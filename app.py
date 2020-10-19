from config import create_app, db
from views import public_route_bp
from controller import api_bp_new, api_bp_old


app = create_app()
"""
register blueprint
"""
app.register_blueprint(public_route_bp)
app.register_blueprint(api_bp_old)
app.register_blueprint(api_bp_new, url_prefix='/api')

app.app_context().push()


def main():
    app.run(debug=True, host="0.0.0.0", port=5000)  # RUNNING APP MAKE debug =FALSE for Production Env
    # app.run(debug=True, host="localhost", port=8888)


if __name__ == "__main__":
    main()
