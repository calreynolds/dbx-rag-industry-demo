# Databricks notebook source
# MAGIC %run ./LakehouseApps-helpers

# COMMAND ----------

list()

# COMMAND ----------

# MAGIC %md
# MAGIC # Commands
# MAGIC
# MAGIC | App APIs    | Notes | Example |
# MAGIC | -------- | ------- | ------- |
# MAGIC | list()   | list all apps in workspace | list() |
# MAGIC | details(app_name) | Get app details, including URL | details("taxis") |
# MAGIC | create(app_name, app_description) | Need to deploy() before the app is usable | create("taxis2", "New York Taxis") |
# MAGIC | deploy(app_name, source_code_path) | Can deploy multiple times to the same app | deploy("taxis2", "/Workspace/Shared/lakehouse-apps/streamlit_app")
# MAGIC | delete(app_name) | Goodbye! | delete("taxis2") |
# MAGIC
# MAGIC See also [Lakehouse Apps Instruction Manual](https://docs.google.com/document/d/1Fl_P7nAhozFD_dTsbEaUDARPttlBXzoYlvVk_FZCw-8/edit?usp=sharing).
# MAGIC

# COMMAND ----------

details("test-ydmao")