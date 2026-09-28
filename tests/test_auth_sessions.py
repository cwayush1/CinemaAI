"""
Unit tests for user sessions and login persistence.
"""

import os
import sys
import unittest
import tempfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.database import (
    init_db, register_user, authenticate_user,
    create_user_session, get_user_by_session_token, delete_user_session
)


class TestAuthSessions(unittest.TestCase):

    def setUp(self):
        # Create a temporary SQLite database for clean isolated testing
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        init_db(self.db_path)

    def tearDown(self):
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except OSError:
                pass

    def test_session_lifecycle(self):
        # 1. Register a test user
        ok, user = register_user("test_persister", "secret123", "test@test.com", "Action|Drama", self.db_path)
        self.assertTrue(ok)
        user_id = user["id"]

        # 2. Authenticate user
        auth_ok, auth_user = authenticate_user("test_persister", "secret123", self.db_path)
        self.assertTrue(auth_ok)
        self.assertEqual(auth_user["id"], user_id)

        # 3. Create session token (simulating login)
        token = create_user_session(user_id, self.db_path)
        self.assertIsNotNone(token)
        self.assertGreater(len(token), 20)

        # 4. Retrieve user using token (simulating page reload F5)
        restored = get_user_by_session_token(token, self.db_path)
        self.assertIsNotNone(restored)
        self.assertEqual(restored["id"], user_id)
        self.assertEqual(restored["username"], "test_persister")
        self.assertEqual(restored["email"], "test@test.com")
        self.assertEqual(restored["preferred_genres"], "Action|Drama")

        # 5. Delete session (simulating sign out)
        delete_user_session(token, self.db_path)
        after_signout = get_user_by_session_token(token, self.db_path)
        self.assertIsNone(after_signout)

    def test_invalid_token(self):
        # Non-existent or empty token returns None
        self.assertIsNone(get_user_by_session_token("non_existent_token_12345", self.db_path))
        self.assertIsNone(get_user_by_session_token("", self.db_path))
        self.assertIsNone(get_user_by_session_token(None, self.db_path))


if __name__ == "__main__":
    unittest.main()
