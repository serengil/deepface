from flask import Blueprint, render_template

public_route_bp = Blueprint('public_route', __name__)


@public_route_bp.route('/')
def render_homepage():
    return render_template("homePage.html")


@public_route_bp.route('/breed-list')
def render_breed_info_page():
    return render_template('breedPage.html')


@public_route_bp.route('/statistical-data')
def render_statistical_page():
    return render_template('statisticalPage.html')