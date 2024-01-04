from python:3.11.6

WORKDIR /dbt

# Install dbt
RUN pip install dbt-postgres

COPY . .
CMD ["dbt", "run"]