from flask import (Response)
from json import dumps


def send_json_response(body, status):
    """
    format json response
    :param body: dic object going to send
    :param status: response status
    :return: formatted response
    """
    return Response(response=dumps(body),
                    status=status,
                    mimetype="application/json")
