from pathlib import Path
import requests
import json

URL_VERIFY = "http://127.0.0.1:5005/verify"
HEADERS = {"Content-Type": "application/json"}

if __name__ == "__main__":
    result = {
        "results": [
            {
                "embedding": [
                    -1.2134509086608887,
                    -1.1512534618377686,
                    1.3474149703979492,
                    -1.4069346189498901,
                    1.5224522352218628,
                    -2.0914249420166016,
                    -0.6764576435089111,
                    0.9096606373786926,
                    -0.9535425305366516,
                    -0.7557652592658997,
                    1.4426558017730713,
                    0.39987713098526,
                    1.0626652240753174,
                    -0.7775546312332153,
                    1.518526554107666,
                    0.516917884349823,
                    0.9891947507858276,
                    -1.9875643253326416,
                    -1.0734516382217407,
                    -0.8696305751800537,
                    -1.6237026453018188,
                    -1.4600988626480103,
                    0.5612819194793701,
                    2.2840688228607178,
                    -0.7382596731185913,
                    -0.5750569105148315,
                    1.4154270887374878,
                    2.2369160652160645,
                    -0.053926825523376465,
                    0.15387092530727386,
                    0.4230121970176697,
                    -2.1653738021850586,
                    0.6751581430435181,
                    1.4729801416397095,
                    -0.15704210102558136,
                    1.2364482879638672,
                    0.7515401244163513,
                    -0.16530320048332214,
                    0.6278983950614929,
                    0.7643095254898071,
                    -1.301100730895996,
                    -0.3106960952281952,
                    -0.24317842721939087,
                    0.4929760694503784,
                    -0.9474270343780518,
                    -0.3621395230293274,
                    -0.2367379069328308,
                    -1.0334841012954712,
                    0.08997400104999542,
                    -0.9825359582901001,
                    -0.55338054895401,
                    0.7562239170074463,
                    -0.22758561372756958,
                    -0.6250701546669006,
                    1.1473084688186646,
                    1.0343632698059082,
                    -0.14173297584056854,
                    0.4334374964237213,
                    0.4255795180797577,
                    1.1381419897079468,
                    -0.11195407062768936,
                    0.274019718170166,
                    1.3193482160568237,
                    0.23514097929000854,
                    -1.0187417268753052,
                    1.4613186120986938,
                    -2.419894218444824,
                    1.9854289293289185,
                    -0.21312934160232544,
                    1.3490840196609497,
                    0.13454176485538483,
                    0.047936052083969116,
                    -1.362741470336914,
                    -0.7827252745628357,
                    0.6984714865684509,
                    2.454777479171753,
                    -1.6840591430664062,
                    0.2888719439506531,
                    -0.28213101625442505,
                    0.0014555901288986206,
                    -1.1008340120315552,
                    1.4643453359603882,
                    -0.3036220967769623,
                    0.2829480767250061,
                    1.3157472610473633,
                    0.2576085925102234,
                    -0.5991222262382507,
                    1.2115517854690552,
                    -1.3916406631469727,
                    -1.0471534729003906,
                    0.6068964004516602,
                    -0.8932938575744629,
                    0.6621257066726685,
                    -0.13733410835266113,
                    0.7743759155273438,
                    0.027307990938425064,
                    -1.8998537063598633,
                    -0.5596081614494324,
                    0.8760293126106262,
                    0.9114910364151001,
                    0.04533162713050842,
                    0.4408164322376251,
                    0.1473987102508545,
                    0.22080418467521667,
                    -0.7411225438117981,
                    0.6423298120498657,
                    -0.6457950472831726,
                    -1.218443512916565,
                    -1.465571403503418,
                    -0.6502959132194519,
                    0.44952529668807983,
                    -0.5342910885810852,
                    0.358688086271286,
                    -1.7372446060180664,
                    0.5452773571014404,
                    1.4514544010162354,
                    0.4862821400165558,
                    -0.10726253688335419,
                    0.11322827637195587,
                    0.404361754655838,
                    1.3982288837432861,
                    0.3022083640098572,
                    1.1285101175308228,
                    -3.507411479949951,
                    -0.5437541007995605,
                    -0.31597772240638733,
                    0.9231274724006653,
                    -0.41314083337783813,
                ],
                "face_confidence": 0.92,
                "facial_area": {
                    "h": 998,
                    "left_eye": [1292, 1300],
                    "right_eye": [912, 1283],
                    "w": 998,
                    "x": 621,
                    "y": 906,
                },
            }
        ]
    }

    vec1 = result["results"][0]["embedding"]
    vec2 = vec1

    payload = {
        "model_name": "GhostFaceNet",
        "img1_path": vec1,
        "img2_path": vec2,
    }

    response = requests.request(
        "POST", URL_VERIFY, data=json.dumps(payload), headers=HEADERS
    )

    print(response.text)
