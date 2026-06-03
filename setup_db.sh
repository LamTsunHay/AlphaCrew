#!/usr/bin/env bash
# setup_db.sh — one-shot PostgreSQL + pgvector setup for Strategy Engine v5.1
# Run as a user with sudo access:  bash setup_db.sh
set -euo pipefail

DB_NAME="alphacrew"
DB_USER="${DB_USER:-user}"
DB_PASS="${DB_PASS:-alphacrew}"
DB_HOST="localhost"
DB_PORT="5432"

echo "=== 1/4  Installing pgvector for PostgreSQL 18 ==="
sudo apt-get install -y postgresql-18-pgvector

echo ""
echo "=== 2/4  Creating role and database ==="

# Create the role (idempotent via DO block — runs inside a transaction)
sudo -u postgres psql -c "
DO \$\$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = '${DB_USER}') THEN
    CREATE ROLE \"${DB_USER}\" LOGIN PASSWORD '${DB_PASS}';
    RAISE NOTICE 'Role created: ${DB_USER}';
  ELSE
    RAISE NOTICE 'Role already exists: ${DB_USER}';
  END IF;
END
\$\$;"

# CREATE DATABASE must run outside a transaction block — use shell guard
DB_EXISTS=$(sudo -u postgres psql -tAc \
  "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'")
if [ "${DB_EXISTS}" = "1" ]; then
  echo "Database '${DB_NAME}' already exists — skipping."
else
  sudo -u postgres psql -c "CREATE DATABASE \"${DB_NAME}\" OWNER \"${DB_USER}\";"
  echo "Database '${DB_NAME}' created."
fi

sudo -u postgres psql -c \
  "GRANT ALL PRIVILEGES ON DATABASE \"${DB_NAME}\" TO \"${DB_USER}\";"

echo ""
echo "=== 3/4  Applying migration: 001_create_setups.sql ==="
PGPASSWORD="${DB_PASS}" psql \
  -h "${DB_HOST}" \
  -p "${DB_PORT}" \
  -U "${DB_USER}" \
  -d "${DB_NAME}" \
  -f "$(dirname "$0")/migrations/001_create_setups.sql"

echo ""
echo "=== 4/4  Writing DATABASE_URL to .env ==="
ENV_FILE="$(dirname "$0")/.env"
DSN="postgresql://${DB_USER}:${DB_PASS}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

if [ -f "${ENV_FILE}" ]; then
  if grep -q "^DATABASE_URL=" "${ENV_FILE}"; then
    sed -i "s|^DATABASE_URL=.*|DATABASE_URL=${DSN}|" "${ENV_FILE}"
    echo "Updated DATABASE_URL in .env"
  else
    echo "DATABASE_URL=${DSN}" >> "${ENV_FILE}"
    echo "Appended DATABASE_URL to .env"
  fi
else
  echo "DATABASE_URL=${DSN}" > "${ENV_FILE}"
  echo "Created .env with DATABASE_URL"
fi

echo ""
echo "Done! Connection string: ${DSN}"
echo ""
echo "Verify with:"
echo "  python -c \"import database; conn = database.initialize_database(); print('OK')\""
