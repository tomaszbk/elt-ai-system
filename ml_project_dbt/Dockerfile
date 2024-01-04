from python:3.11.6

WORKDIR /dbt

# Install dbt
RUN pip install dbt-postgres

COPY profiles.yml ~/.dbt/profiles.yml
COPY . .

CMD ["dbt", "run"]