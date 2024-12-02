# Example content
echo "Starting the application..."
exec "$@"

gunicorn --workers=1 --timeout=7200 --bind=0.0.0.0:5000 --log-level=debug --access-logformat='%(h)s - - [%(t)s] "%(r)s" %(s)s %(b)s %(L)s' --access-logfile=- "app:create_app()"
