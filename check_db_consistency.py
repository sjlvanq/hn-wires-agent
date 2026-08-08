#!/usr/bin/env python3
"""
Script to check the consistency of database relations.

Checks:
1. Each keyword must have an embedding
2. Each wire must have an embedding
3. Each wire must point to an existing post
4. Each keyword must point to an existing post
5. Each bookmark must point to an existing post
6. Each post_processing_state must point to an existing post

Usage:
    python check_db_consistency.py              # Check only
    python check_db_consistency.py --fix        # Check and fix issues
"""

import sys
import os
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from config.settings import settings
from database.connection import DatabaseConnection


class DatabaseConsistencyChecker:
    def __init__(self):
        self.db = DatabaseConnection()
        self.issues = []
        self.vec_available = True  # sqlite_vec is always available with DatabaseConnection

    def check_keyword_embeddings(self):
        """Check that each keyword has an embedding."""
        print("Checking keyword embeddings...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all keyword IDs
            cursor.execute("SELECT id, post_id, keyword FROM keywords;")
            keywords = cursor.fetchall()

            # Get all keyword embedding rowids
            cursor.execute("SELECT rowid FROM vec_keywords;")
            embedding_rowids = set(row[0] for row in cursor.fetchall())

            # Find keywords without embeddings
            missing_embeddings = []
            for keyword_id, post_id, keyword in keywords:
                if keyword_id not in embedding_rowids:
                    missing_embeddings.append({
                        'keyword_id': keyword_id,
                        'post_id': post_id,
                        'keyword': keyword
                    })

            if missing_embeddings:
                self.issues.append({
                    'type': 'missing_keyword_embeddings',
                    'count': len(missing_embeddings),
                    'details': missing_embeddings
                })
                print(f"  ❌ Found {len(missing_embeddings)} keywords without embeddings")
            else:
                print(f"  ✓ All {len(keywords)} keywords have embeddings")

    def check_wire_embeddings(self):
        """Check that each wire has an embedding."""
        print("Checking wire embeddings...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all wire IDs
            cursor.execute("SELECT id, post_id FROM wires;")
            wires = cursor.fetchall()

            # Get all wire embedding rowids
            cursor.execute("SELECT rowid FROM vec_wires;")
            embedding_rowids = set(row[0] for row in cursor.fetchall())

            # Find wires without embeddings
            missing_embeddings = []
            for wire_id, post_id in wires:
                if wire_id not in embedding_rowids:
                    missing_embeddings.append({
                        'wire_id': wire_id,
                        'post_id': post_id
                    })

            if missing_embeddings:
                self.issues.append({
                    'type': 'missing_wire_embeddings',
                    'count': len(missing_embeddings),
                    'details': missing_embeddings
                })
                print(f"  ❌ Found {len(missing_embeddings)} wires without embeddings")
            else:
                print(f"  ✓ All {len(wires)} wires have embeddings")

    def check_wire_post_references(self):
        """Check that each wire points to an existing post."""
        print("Checking wire post references...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all wire post_ids
            cursor.execute("SELECT id, post_id FROM wires;")
            wires = cursor.fetchall()

            # Get all existing post IDs
            cursor.execute("SELECT id FROM posts;")
            post_ids = set(row[0] for row in cursor.fetchall())

            # Find wires pointing to non-existent posts
            invalid_references = []
            for wire_id, post_id in wires:
                if post_id not in post_ids:
                    invalid_references.append({
                        'wire_id': wire_id,
                        'post_id': post_id
                    })

            if invalid_references:
                self.issues.append({
                    'type': 'invalid_wire_post_references',
                    'count': len(invalid_references),
                    'details': invalid_references
                })
                print(f"  ❌ Found {len(invalid_references)} wires pointing to non-existent posts")
            else:
                print(f"  ✓ All {len(wires)} wires point to existing posts")

    def check_keyword_post_references(self):
        """Check that each keyword points to an existing post."""
        print("Checking keyword post references...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all keyword post_ids
            cursor.execute("SELECT id, post_id, keyword FROM keywords;")
            keywords = cursor.fetchall()

            # Get all existing post IDs
            cursor.execute("SELECT id FROM posts;")
            post_ids = set(row[0] for row in cursor.fetchall())

            # Find keywords pointing to non-existent posts
            invalid_references = []
            for keyword_id, post_id, keyword in keywords:
                if post_id not in post_ids:
                    invalid_references.append({
                        'keyword_id': keyword_id,
                        'post_id': post_id,
                        'keyword': keyword
                    })

            if invalid_references:
                self.issues.append({
                    'type': 'invalid_keyword_post_references',
                    'count': len(invalid_references),
                    'details': invalid_references
                })
                print(f"  ❌ Found {len(invalid_references)} keywords pointing to non-existent posts")
            else:
                print(f"  ✓ All {len(keywords)} keywords point to existing posts")

    def check_bookmark_post_references(self):
        """Check that each bookmark points to an existing post."""
        print("Checking bookmark post references...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all bookmark post_ids
            cursor.execute("SELECT id, post_id FROM bookmarks;")
            bookmarks = cursor.fetchall()

            # Get all existing post IDs
            cursor.execute("SELECT id FROM posts;")
            post_ids = set(row[0] for row in cursor.fetchall())

            # Find bookmarks pointing to non-existent posts
            invalid_references = []
            for bookmark_id, post_id in bookmarks:
                if post_id not in post_ids:
                    invalid_references.append({
                        'bookmark_id': bookmark_id,
                        'post_id': post_id
                    })

            if invalid_references:
                self.issues.append({
                    'type': 'invalid_bookmark_post_references',
                    'count': len(invalid_references),
                    'details': invalid_references
                })
                print(f"  ❌ Found {len(invalid_references)} bookmarks pointing to non-existent posts")
            else:
                print(f"  ✓ All {len(bookmarks)} bookmarks point to existing posts")

    def check_post_processing_state_references(self):
        """Check that each post_processing_state points to an existing post."""
        print("Checking post_processing_state post references...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all post_processing_state post_ids
            cursor.execute("SELECT post_id FROM post_processing_state;")
            processing_states = cursor.fetchall()

            # Get all existing post IDs
            cursor.execute("SELECT id FROM posts;")
            post_ids = set(row[0] for row in cursor.fetchall())

            # Find post_processing_state pointing to non-existent posts
            invalid_references = []
            for (post_id,) in processing_states:
                if post_id not in post_ids:
                    invalid_references.append({
                        'post_id': post_id
                    })

            if invalid_references:
                self.issues.append({
                    'type': 'invalid_post_processing_state_references',
                    'count': len(invalid_references),
                    'details': invalid_references
                })
                print(f"  ❌ Found {len(invalid_references)} post_processing_state entries pointing to non-existent posts")
            else:
                print(f"  ✓ All {len(processing_states)} post_processing_state entries point to existing posts")

    def fix_missing_keyword_embeddings(self):
        """Delete keywords that don't have embeddings."""
        print("Fixing missing keyword embeddings...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all keyword IDs that have embeddings
            cursor.execute("SELECT rowid FROM vec_keywords;")
            embedding_rowids = set(row[0] for row in cursor.fetchall())

            # Delete keywords without embeddings
            cursor.execute("SELECT id FROM keywords;")
            all_keyword_ids = [row[0] for row in cursor.fetchall()]

            keywords_to_delete = [kid for kid in all_keyword_ids if kid not in embedding_rowids]

            if keywords_to_delete:
                cursor.executemany("DELETE FROM keywords WHERE id = ?",
                                   [(kid,) for kid in keywords_to_delete])
                conn.commit()
                print(f"  ✓ Deleted {len(keywords_to_delete)} keywords without embeddings")
                return len(keywords_to_delete)
            else:
                print("  ✓ No keywords to delete")
                return 0

    def fix_missing_wire_embeddings(self):
        """Delete wires that don't have embeddings."""
        print("Fixing missing wire embeddings...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all wire IDs that have embeddings
            cursor.execute("SELECT rowid FROM vec_wires;")
            embedding_rowids = set(row[0] for row in cursor.fetchall())

            # Delete wires without embeddings
            cursor.execute("SELECT id FROM wires;")
            all_wire_ids = [row[0] for row in cursor.fetchall()]

            wires_to_delete = [wid for wid in all_wire_ids if wid not in embedding_rowids]

            if wires_to_delete:
                cursor.executemany("DELETE FROM wires WHERE id = ?",
                                   [(wid,) for wid in wires_to_delete])
                conn.commit()
                print(f"  ✓ Deleted {len(wires_to_delete)} wires without embeddings")
                return len(wires_to_delete)
            else:
                print("  ✓ No wires to delete")
                return 0

    def fix_invalid_wire_post_references(self):
        """Delete wires that point to non-existent posts."""
        print("Fixing invalid wire post references...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all existing post IDs
            cursor.execute("SELECT id FROM posts;")
            post_ids = set(row[0] for row in cursor.fetchall())

            # Get wires with invalid post references
            cursor.execute("SELECT id, post_id FROM wires;")
            wires = cursor.fetchall()

            wires_to_delete = [wire_id for wire_id, post_id in wires if post_id not in post_ids]

            if wires_to_delete:
                # First delete their vector embeddings
                cursor.executemany("DELETE FROM vec_wires WHERE rowid = ?",
                                   [(wid,) for wid in wires_to_delete])

                # Then delete the wires
                cursor.executemany("DELETE FROM wires WHERE id = ?",
                                   [(wid,) for wid in wires_to_delete])
                conn.commit()
                print(f"  ✓ Deleted {len(wires_to_delete)} wires with invalid post references")
                return len(wires_to_delete)
            else:
                print("  ✓ No wires to delete")
                return 0

    def fix_invalid_keyword_post_references(self):
        """Delete keywords that point to non-existent posts."""
        print("Fixing invalid keyword post references...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all existing post IDs
            cursor.execute("SELECT id FROM posts;")
            post_ids = set(row[0] for row in cursor.fetchall())

            # Get keywords with invalid post references
            cursor.execute("SELECT id, post_id FROM keywords;")
            keywords = cursor.fetchall()

            keywords_to_delete = [keyword_id for keyword_id, post_id in keywords if post_id not in post_ids]

            if keywords_to_delete:
                # First delete their vector embeddings
                cursor.executemany("DELETE FROM vec_keywords WHERE rowid = ?",
                                   [(kid,) for kid in keywords_to_delete])

                # Then delete the keywords
                cursor.executemany("DELETE FROM keywords WHERE id = ?",
                                   [(kid,) for kid in keywords_to_delete])
                conn.commit()
                print(f"  ✓ Deleted {len(keywords_to_delete)} keywords with invalid post references")
                return len(keywords_to_delete)
            else:
                print("  ✓ No keywords to delete")
                return 0

    def fix_invalid_bookmark_post_references(self):
        """Delete bookmarks that point to non-existent posts."""
        print("Fixing invalid bookmark post references...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all existing post IDs
            cursor.execute("SELECT id FROM posts;")
            post_ids = set(row[0] for row in cursor.fetchall())

            # Get bookmarks with invalid post references
            cursor.execute("SELECT id, post_id FROM bookmarks;")
            bookmarks = cursor.fetchall()

            bookmarks_to_delete = [bookmark_id for bookmark_id, post_id in bookmarks if post_id not in post_ids]

            if bookmarks_to_delete:
                cursor.executemany("DELETE FROM bookmarks WHERE id = ?",
                                   [(bid,) for bid in bookmarks_to_delete])
                conn.commit()
                print(f"  ✓ Deleted {len(bookmarks_to_delete)} bookmarks with invalid post references")
                return len(bookmarks_to_delete)
            else:
                print("  ✓ No bookmarks to delete")
                return 0

    def fix_invalid_post_processing_state_references(self):
        """Delete post_processing_state entries that point to non-existent posts."""
        print("Fixing invalid post_processing_state references...")

        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Get all existing post IDs
            cursor.execute("SELECT id FROM posts;")
            post_ids = set(row[0] for row in cursor.fetchall())

            # Get post_processing_state with invalid post references
            cursor.execute("SELECT post_id FROM post_processing_state;")
            processing_states = cursor.fetchall()

            states_to_delete = [post_id for (post_id,) in processing_states if post_id not in post_ids]

            if states_to_delete:
                cursor.executemany("DELETE FROM post_processing_state WHERE post_id = ?",
                                   [(pid,) for pid in states_to_delete])
                conn.commit()
                print(f"  ✓ Deleted {len(states_to_delete)} post_processing_state entries with invalid post references")
                return len(states_to_delete)
            else:
                print("  ✓ No post_processing_state entries to delete")
                return 0

    def fix_all_issues(self):
        """Fix all detected issues."""
        print("\n" + "="*60)
        print("FIXING DATABASE ISSUES")
        print("="*60)

        total_fixed = 0

        for issue in self.issues:
            issue_type = issue['type']
            print(f"\nFixing: {issue_type}")

            if issue_type == 'missing_keyword_embeddings':
                fixed = self.fix_missing_keyword_embeddings()
            elif issue_type == 'missing_wire_embeddings':
                fixed = self.fix_missing_wire_embeddings()
            elif issue_type == 'invalid_wire_post_references':
                fixed = self.fix_invalid_wire_post_references()
            elif issue_type == 'invalid_keyword_post_references':
                fixed = self.fix_invalid_keyword_post_references()
            elif issue_type == 'invalid_bookmark_post_references':
                fixed = self.fix_invalid_bookmark_post_references()
            elif issue_type == 'invalid_post_processing_state_references':
                fixed = self.fix_invalid_post_processing_state_references()
            else:
                print(f"  ⚠ Unknown issue type: {issue_type}")
                fixed = 0

            total_fixed += fixed

        print("\n" + "="*60)
        print(f"Total records deleted: {total_fixed}")
        print("="*60)

        return total_fixed

    def print_summary(self):
        """Print a summary of all issues found."""
        print("\n" + "="*60)
        print("CONSISTENCY CHECK SUMMARY")
        print("="*60)
        
        if not self.issues:
            print("✓ All consistency checks passed!")
            return True
        
        print(f"❌ Found {len(self.issues)} type(s) of issues:\n")
        
        for issue in self.issues:
            print(f"Issue Type: {issue['type']}")
            print(f"Count: {issue['count']}")
            
            if issue['count'] <= 10:
                print("Details:")
                for detail in issue['details']:
                    print(f"  - {detail}")
            else:
                print(f"Details: (showing first 10 of {issue['count']})")
                for detail in issue['details'][:10]:
                    print(f"  - {detail}")
                print(f"  ... and {issue['count'] - 10} more")
            print()
        
        return False

    def run_all_checks(self):
        """Run all consistency checks."""
        print("Starting database consistency checks...")
        print("="*60)

        try:
            self.check_keyword_embeddings()
            self.check_wire_embeddings()
            self.check_wire_post_references()
            self.check_keyword_post_references()
            self.check_bookmark_post_references()
            self.check_post_processing_state_references()

            return self.print_summary()

        except Exception as e:
            print(f"❌ Database error: {e}")
            return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check and fix database consistency issues')
    parser.add_argument('--fix', action='store_true', 
                        help='Automatically fix detected issues by deleting orphan rows')
    parser.add_argument('--yes', action='store_true',
                        help='Skip confirmation prompt when fixing')
    
    args = parser.parse_args()
    
    checker = DatabaseConsistencyChecker()
    success = checker.run_all_checks()
    
    # If issues were found and --fix flag is set
    if not success and args.fix:
        if not args.yes:
            print("\n" + "="*60)
            response = input("Do you want to fix these issues by deleting orphan rows? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Skipping fixes.")
                sys.exit(1)
        
        total_fixed = checker.fix_all_issues()
        
        # Re-run checks to verify fixes
        print("\n" + "="*60)
        print("RE-RUNNING CHECKS AFTER FIXES")
        print("="*60)
        checker.issues = []  # Clear previous issues
        success = checker.run_all_checks()
        
        if success:
            print("\n✓ Database is now consistent!")
            sys.exit(0)
        else:
            print("\n⚠ Some issues remain after fixes.")
            sys.exit(1)
    else:
        sys.exit(0 if success else 1)
