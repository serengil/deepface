# Example content
echo "Starting the application..."
exec "$@"

# if you plan to serve deepface over grpc, then activate this line and generate grpc stubs in Dockerfile
# python grpc_server.py &

gunicorn --workers=1 --timeout=7200 --bind=0.0.0.0:5000 --log-level=debug --access-logformat='%(h)s - - [%(t)s] "%(r)s" %(s)s %(b)s %(L)s' --access-logfile=- "app:create_app()"
