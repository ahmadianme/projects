from flask import Flask, jsonify, make_response
from flask_restful import Resource, Api, reqparse
from flask_cors import CORS
from model import Model
import json

app = Flask(__name__)
api = Api(app)

CORS(app, origins=['http://localhost:8888'])

# @app.teardown_appcontext
# def teardown_db(exception):
#     db = getattr(g, '_database', None)
#     if db is not None:
#         db.close()

model = Model()
model.start()

@app.route("/")
def index():
    return 'Welcome to AI inference by DayanSystem (www.dayansystem.com).<br> AI service is ready...', 200


class AiInferenceApi(Resource):
    def post(self):
        parser = reqparse.RequestParser()

        parser.add_argument('inputs', required=True, type=dict, location="json")

        # Parse the arguments into an object
        args = parser.parse_args()

        # print(args)

        responseData = {}

        try:
            responseData['modelOutput'] = model.predict(args.inputs)
            responseData['status'] = 'success'

            print(responseData)



            response = make_response(json.dumps(responseData), 200)
            # response.headers['Access-Control-Allow-Origin'] = 'http://localhost:8888'
            # response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            # response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
            # response.headers['Access-Control-Allow-Methods'] = 'POST'
            return response
        except Exception as error:
            responseData['status'] = 'error'
            responseData['systemMessage'] = [{
                'text': 'An error occurred while processing request.'
            }]
            responseData['message'] = [{
                'type': 'danger',
                'text': 'خطایی در هنگام پردازش درخواست شما رخ داده است.'
            }]

            print(error)

            response = make_response(json.dumps(responseData), 500)
            response.headers['Access-Control-Allow-Origin'] = 'http://localhost:8888'
            return response



api.add_resource(AiInferenceApi, '/api/nash/predict')



