# Example content
echo "Starting the application..."
exec "$@"

gunicorn -w 1 -b 0.0.0.0:5000 --timeout 7200 --log-level 'debug' --access-logfile - --access-logformat '%(h)s - - [%(t)s] "%(r)s" %(s)s %(b)s %(L)s' app:create_app