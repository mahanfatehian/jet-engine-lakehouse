-- =============================================================================
-- Least-privilege roles of the operational database.
--
-- Applied by the one-shot `postgres-init` compose service on every `up`, because
-- init.sql only runs on an empty volume: a stack created before this file gets
-- the roles too, and a changed password or grant applies on the next `up`. Every
-- statement is idempotent, and the roles change in one transaction.
--
--   POSTGRES_CDC_USER    Debezium: LOGIN REPLICATION and SELECT on the three CDC
--                        tables (initial snapshot); it streams through the
--                        publication of init.sql and its replication slot
--   POSTGRES_AUDIT_USER  the reconciliation audit, through ClickHouse's
--                        postgresql(): SELECT on engine_telemetry, nothing else
--
-- psql takes the names and passwords from the environment (\getenv, psql 15) and
-- quotes them itself (:"name" as an identifier, :'value' as a literal); it echoes
-- no statement, and the session keeps the server from logging one.
-- =============================================================================
\set ON_ERROR_STOP on
\getenv cdc_user POSTGRES_CDC_USER
\getenv cdc_password POSTGRES_CDC_PASSWORD
\getenv audit_user POSTGRES_AUDIT_USER
\getenv audit_password POSTGRES_AUDIT_PASSWORD

-- ALTER ROLE ... PASSWORD would otherwise reach the server log with log_statement, a duration or
-- sampling threshold, or an error.
SET log_statement = 'none';
SET log_min_error_statement = 'panic';
SET log_min_duration_statement = -1;
SET log_min_duration_sample = -1;
SET log_transaction_sample_rate = 0;

-- Three distinct roles: otherwise the statements below would demote the superuser this file
-- runs as, or turn the replication role into the audit role.
SELECT :'cdc_user' IN (current_user, :'audit_user') OR :'audit_user' = current_user AS roles_clash \gset
\if :roles_clash
DO $$ BEGIN
    RAISE EXCEPTION 'POSTGRES_CDC_USER, POSTGRES_AUDIT_USER and POSTGRES_USER must name three roles';
END $$;
\endif

BEGIN;

SELECT format('CREATE ROLE %I', :'cdc_user')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'cdc_user') \gexec
ALTER ROLE :"cdc_user" WITH LOGIN REPLICATION NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS
    PASSWORD :'cdc_password';

SELECT format('CREATE ROLE %I', :'audit_user')
WHERE NOT EXISTS (SELECT FROM pg_roles WHERE rolname = :'audit_user') \gexec
ALTER ROLE :"audit_user" WITH LOGIN NOREPLICATION NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS
    PASSWORD :'audit_password';

-- Revoked first, so a grant this file no longer makes disappears on the next `up`.
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM :"cdc_user", :"audit_user";
GRANT SELECT ON engine_status_registry, maintenance_log, engine_telemetry TO :"cdc_user";
GRANT SELECT ON engine_telemetry TO :"audit_user";
-- PUBLIC may create temporary tables by default; neither role writes anything.
REVOKE TEMPORARY ON DATABASE :"DBNAME" FROM PUBLIC;

COMMIT;

-- What `up` left in place, one line per role (no passwords).
SELECT rolname AS role, rolreplication AS replication,
       (SELECT string_agg(relname, ', ' ORDER BY relname)
        FROM pg_class
        WHERE relnamespace = 'public'::regnamespace AND relkind = 'r'
          AND has_table_privilege(pg_roles.oid, pg_class.oid, 'SELECT')) AS select_on
FROM pg_roles
WHERE rolname IN (:'cdc_user', :'audit_user')
ORDER BY rolname;
