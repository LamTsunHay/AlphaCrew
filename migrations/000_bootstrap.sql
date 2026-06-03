-- migrations/000_bootstrap.sql
-- Run once as postgres superuser to bootstrap the database and role.
-- Safe to re-run (all statements are idempotent).

-- 1. Create the application role
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'postgresql') THEN
    CREATE ROLE postgresql LOGIN PASSWORD 'admin';
    RAISE NOTICE 'Role "postgresql" created.';
  ELSE
    RAISE NOTICE 'Role "postgresql" already exists.';
  END IF;
END
$$;

-- 2. Create the database (must run outside a transaction block — see shell command below)
-- This file handles only the role; the DB is created by the shell script.
