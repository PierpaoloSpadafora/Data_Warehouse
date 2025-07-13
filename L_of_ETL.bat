
@echo off
set PGPASSWORD=postgres
psql -U postgres -d DataW -f ./PostgreSQL/1_Creazione_Schema.sql

cd PostgreSQL
python 2_Load_CSV_in_Schema.py
cd ..