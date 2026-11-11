"""User values and constraint names must not masquerade as native duplicate errors."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from deepface.modules.database.mssql import MsSqlClient, _is_unique_violation


class IntegrityError(Exception):
    pass


@pytest.mark.parametrize("code", [2601, 2627])
@pytest.mark.parametrize(
    "suffix",
    ["", " (SQLExecDirectW)", " (SQLExecute); [23000] Statement terminated. (3621)"],
)
def test_native_duplicate_diagnostics_are_recognized(code, suffix):
    assert _is_unique_violation(
        IntegrityError("23000", f"[SQL Server]Duplicate key. ({code}){suffix}")
    )


@pytest.mark.parametrize(
    "message",
    [
        "[SQL Server]CHECK constraint 'CK_2627' rejected the value. (547) (SQLExecDirectW)",
        "[SQL Server]Cannot insert NULL into column 'value2601'. (515) (SQLExecute)",
        "[SQL Server]CHECK constraint 'not (2627) a duplicate' failed. (547)",
    ],
)
def test_unrelated_constraint_failures_are_not_duplicates(message):
    assert not _is_unique_violation(IntegrityError("23000", message))


def test_registration_propagates_the_original_nonduplicate_integrity_error():
    error = IntegrityError(
        "23000", "[SQL Server]CHECK constraint 'CK_2627' failed. (547) (SQLExecDirectW)"
    )
    client = MsSqlClient.__new__(MsSqlClient)
    client.pyodbc = SimpleNamespace(IntegrityError=IntegrityError)
    client.conn = Mock()
    cursor = client.conn.cursor.return_value
    cursor.executemany.side_effect = error
    row = dict(
        img_name="image.png",
        face=np.zeros((2, 2, 3)),
        embedding=[1.0, 0.5],
        model_name="VGG-Face",
        detector_backend="opencv",
        aligned=True,
        l2_normalized=False,
    )
    with pytest.raises(IntegrityError) as raised:
        client.insert_embeddings([row])
    assert raised.value is error
    client.conn.rollback.assert_called_once()
    cursor.close.assert_called_once()
