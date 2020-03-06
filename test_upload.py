import contextlib
import unittest
from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from string import Formatter
from typing import Any, ContextManager, Dict, List, NamedTuple, Optional, Union

import pyarrow
from numpy.testing import assert_equal

from cjwparse._util import tempfile_context
from cjwparse.testing.i18n import cjwparse_i18n_message
from upload import render as upload

RenderResult = namedtuple("RenderResult", ["table", "errors"])


class ColumnType(ABC):
    """
    Data type of a column.

    This describes how it is presented -- not how its bytes are arranged.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        The name of the type: 'text', 'number' or 'datetime'.
        """
        pass


class Column(NamedTuple):
    """
    A column definition.
    """

    name: str
    """Name of the column."""

    type: ColumnType
    """How the column data is stored and displayed to the user."""


@dataclass(frozen=True)
class ColumnTypeText(ColumnType):
    # override
    @property
    def name(self) -> str:
        return "text"


class NumberFormatter:
    """
    Utility to convert int and float to str.

    Usage:

        formatter = NumberFormatter('${:,.2f}')
        formatter.format(1234.56)  # => "$1,234.56"

    This is similar to Python `format()` but different:

    * It allows formatting float as int: `NumberFormatter('{:d}').format(0.1)`
    * It disallows "conversions" (e.g., `{!r:s}`)
    * It disallows variable name/numbers (e.g., `{1:d}`, `{value:d}`)
    * It raises ValueError on construction if format is imperfect
    * Its `.format()` method always succeeds
    """

    _IntTypeSpecifiers = set("bcdoxXn")
    """
    Type names that operate on integer (as opposed to float).

    Python `format()` auto-converts int to float, but it doesn't auto-convert
    float to int. Workbench does auto-convert float to int: any format that
    works for one Number must work for all Numbers.
    """

    def __init__(self, format_s: str):
        if not isinstance(format_s, str):
            raise ValueError("Format must be str")

        # parts: a list of (literal_text, field_name, format_spec, conversion)
        #
        # The "literal_text" always comes _before_ the field. So we end up
        # with three possibilities:
        #
        #    "prefix{}suffix": [(prefix, "", "", ""), (suffix, None...)]
        #    "prefix{}": [(prefix, "", "", '")]
        #    "{}suffix": [("", "", "", ""), (suffix, None...)]
        parts = list(Formatter().parse(format_s))

        if len(parts) > 2 or len(parts) == 2 and parts[1][1] is not None:
            raise ValueError("Can only format one number")

        if not parts or parts[0][1] is None:
            raise ValueError('Format must look like "{:...}"')

        if parts[0][1] != "":
            raise ValueError("Field names or numbers are not allowed")

        if parts[0][3] is not None:
            raise ValueError("Field converters are not allowed")

        self._prefix = parts[0][0]
        self._format_spec = parts[0][2]
        if len(parts) == 2:
            self._suffix = parts[1][0]
        else:
            self._suffix = ""
        self._need_int = (
            self._format_spec and self._format_spec[-1] in self._IntTypeSpecifiers
        )

        # Test it!
        #
        # A reading of cpython 3.7 Python/formatter_unicode.c
        # parse_internal_render_format_spec() suggests the following unobvious
        # details:
        #
        # * Python won't parse a format spec unless you're formatting a number
        # * _PyLong_FormatAdvancedWriter() accepts a superset of the formats
        #   _PyFloat_FormatAdvancedWriter() accepts. (Workbench accepts that
        #   superset.)
        #
        # Therefore, if we can format an int, the format is valid.
        format(1, self._format_spec)

    def format(self, value: Union[int, float]) -> str:
        if self._need_int:
            value = int(value)
        else:
            # Format float64 _integers_ as int. For instance, '3.0' should be
            # formatted as though it were the int, '3'.
            #
            # Python would normally format '3.0' as '3.0' by default; that's
            # not acceptable to us because we can't write a JavaScript
            # formatter that would do the same thing. (Javascript doesn't
            # distinguish between float and int.)
            int_value = int(value)
            if int_value == value:
                value = int_value

        return self._prefix + format(value, self._format_spec) + self._suffix


@dataclass(frozen=True)
class ColumnTypeNumber(ColumnType):
    # https://docs.python.org/3/library/string.html#format-specification-mini-language
    format: str = "{:,}"  # Python format() string -- default adds commas
    # TODO handle locale, too: format depends on it. Python will make this
    # difficult because it can't format a string in an arbitrary locale: it can
    # only do it using global variables, which we can't use.

    def __post_init__(self):
        formatter = NumberFormatter(self.format)  # raises ValueError
        object.__setattr__(self, "_formatter", formatter)

    # override
    @property
    def name(self) -> str:
        return "number"


@dataclass(frozen=True)
class ColumnTypeDatetime(ColumnType):
    # # https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior
    # format: str = '{}'  # Python format() string

    # # TODO handle locale, too: format depends on it. Python will make this
    # # difficult because it can't format a string in an arbitrary locale: it can
    # # only do it using global variables, which we can't use.

    # override
    @property
    def name(self) -> str:
        return "datetime"


class TableMetadata(NamedTuple):
    """Table data that will be cached for easy access."""

    n_rows: int = 0
    """Number of rows in the table."""

    columns: List[Column] = []
    """Columns -- the user-visible aspects of them, at least."""


def assert_arrow_table_equals(
    result1: pyarrow.Table, result2: Union[Dict[str, Any], pyarrow.Table]
) -> None:
    if isinstance(result2, dict):
        result2 = pyarrow.table(result2)
    assertEqual = unittest.TestCase().assertEqual
    assertEqual(result1.shape, result2.shape)
    assertEqual(result1.column_names, result2.column_names)
    for colname, actual_col, expected_col in zip(
        result1.column_names, result1.columns, result2.columns
    ):
        assertEqual(
            actual_col.type, expected_col.type, msg=f"Column {colname} has wrong type"
        )
        assert_equal(
            actual_col.to_pylist(),
            expected_col.to_pylist(),
            err_msg=f"Column {colname} has wrong values",
        )


@dataclass(frozen=True)
class ArrowTable:
    """
    Table on disk, opened and mmapped.

    A table with no rows must have a file on disk. A table with no _columns_
    is a special case: it _may_ have `table is None and path is None`, or it
    may have an empty Arrow table on disk.

    `self.table` will be populated and validated during construction.

    To pass an ArrowTable between processes, the file must be readable at the
    same `path` to both processes. If your ArrowTable isn't being shared
    between processes, you may safely delete the file at `path` immediately
    after constructing the ArrowTable.
    """

    path: Optional[Path] = None
    """
    Name of file on disk that contains data.

    If the table has columns, the file must exist.
    """

    table: Optional[pyarrow.Table] = None
    """
    Pyarrow table, loaded with mmap.

    If the table has columns, `table` must exist.
    """

    metadata: TableMetadata = field(default_factory=TableMetadata)
    """
    Metadata that agrees with `table`.

    If `table is None`, then `metadata` has no columns.
    """

    @classmethod
    def from_arrow_file_with_inferred_metadata(cls, path: Path) -> "ArrowTable":
        """
        Build from a trusted Arrow file and infer metadata.

        TODO move this function elsewhere.
        """
        # If path does not exist or is empty file, empty ArrowTable
        try:
            if path.stat().st_size == 0:
                return cls()
        except FileNotFoundError:
            return cls()

        with pyarrow.ipc.open_file(path) as reader:
            schema = reader.schema

            # if table has no columns, empty ArrowTable
            columns = [
                Column(name, _pyarrow_type_to_column_type(dtype))
                for name, dtype in zip(schema.names, schema.types)
            ]
            if not columns:
                return cls()

            table = reader.read_all()
        n_rows = table.num_rows
        return cls(path, table, TableMetadata(n_rows, columns))


def _pyarrow_type_to_column_type(dtype: pyarrow.DataType) -> ColumnType:
    if pyarrow.types.is_floating(dtype) or pyarrow.types.is_integer(dtype):
        return ColumnTypeNumber()
    elif pyarrow.types.is_string(dtype) or (
        pyarrow.types.is_dictionary(dtype) and pyarrow.types.is_string(dtype.value_type)
    ):
        return ColumnTypeText()
    elif pyarrow.types.is_timestamp(dtype):
        return ColumnTypeDatetime()
    else:
        return ValueError("Unknown pyarrow type %r" % dtype)


# See UploadFileViewTests for that
class UploadTest(unittest.TestCase):
    @contextlib.contextmanager
    def render(self, params):
        with tempfile_context(prefix="output-", suffix=".arrow") as output_path:
            errors = upload(ArrowTable(), params, output_path)
            table = ArrowTable.from_arrow_file_with_inferred_metadata(output_path)
            yield RenderResult(table, errors)

    @contextlib.contextmanager
    def _file(self, b: bytes, *, suffix) -> ContextManager[Path]:
        with tempfile_context(suffix=suffix) as path:
            path.write_bytes(b)
            yield path

    def test_render_no_file(self):
        with self.render({"file": None, "has_header": True}) as result:
            assert_arrow_table_equals(result.table, {})
            self.assertEqual(result.errors, [])

    def test_render_success(self):
        with self._file(b"A,B\nx,y", suffix=".csv") as path:
            with self.render({"file": path, "has_header": True}) as result:
                assert_arrow_table_equals(result.table, {"A": ["x"], "B": ["y"]})
                self.assertEqual(result.errors, [])

    def test_render_error(self):
        with self._file(b"A,B\nx,y", suffix=".json") as path:
            with self.render({"file": path, "has_header": True}) as result:
                assert_arrow_table_equals(result.table, {})
                self.assertEqual(
                    result.errors,
                    [
                        cjwparse_i18n_message(
                            "TODO_i18n",
                            {"text": "JSON parse error at byte 0: Invalid value."},
                        )
                    ],
                )
