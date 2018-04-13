from flask import Flask, request
from flask_restful import Resource, Api
from sqlalchemy import create_engine
from json import dumps
from flask.ext.jsonpify import jsonify
from datetime import datetime as dt
import numpy as np

db_connect = create_engine('sqlite:///../data/predictions/stocks/predictions.db')
app = Flask(__name__)
api = Api(app)
class StockInfo(Resource):
	def get(self):       
		conn = db_connect.connect()
		query = conn.execute("select * from stockinfo")
		tmp =  [i[1:] for i in query.fetchall()]
		names = [_x[0] for _x in tmp ]
		cols= query.keys()[1:]

		result= {names[j]:{cols[i]:tmp[j][i] for i in range(len(cols))} for j in range(len(names))}

		conn.close()
		return jsonify(result)

class Stock_Prediction_Day(Resource):

	def get(self,stock_label,predictiondate):

		PredictionDay = dt.strptime(predictiondate,'%Y-%m-%d').date()
		
		conn = db_connect.connect()
		query = conn.execute("select * from prediction3BT where label = ? and predictionday = ?",(stock_label,PredictionDay))

		tmp = query.fetchone()[3:]

		cols = query.keys()[3:]
		
		conn.close()
		return jsonify({cols[i]:tmp[i] for i in range(len(cols))})

class Stock_Prediction_Day_All(Resource):

	def get(self,stock_label):

		conn = db_connect.connect()
		
		query = conn.execute("select * from prediction3BT where label = ?",(stock_label,))

		cols = query.keys()

		tmp = query.cursor.fetchall()

		conn.close()

		return jsonify({"PredictionDay "+str(_x[3]):{cols[i]:_x[i] for i in range(3,len(cols))} for _x in tmp})

class Single_Stock_Info(Resource):

	def get(self,stock_label):
		conn = db_connect.connect()

		query = conn.execute("select * from stockinfo where label = ?",(stock_label))

		tmp =  query.fetchone()[1:]

		cols = query.keys()[1:]

		conn.close()

		return jsonify({cols[i]:tmp[i] for i in range(len(cols))})


class Top_Predictions(Resource):

	def get(self,predictiondate):
		PredictionDay = dt.strptime(predictiondate,'%Y-%m-%d').date()

		conn = db_connect.connect()

		query = conn.execute("select label,probcat0,probcat1,probcat2,probcat3,probcat4,probcat5,probcat6,probcat7,probcat8,probcat9,probcat10,probcat11 from prediction3BT where predictionday = ?",(PredictionDay,))
		
		tmp = query.cursor.fetchall()
		labels= np.array([_tmp[0] for _tmp in tmp])
		probs = np.array([_tmp[1:] for _tmp in tmp])
		prob_arg_max = np.argmax(probs,axis=1)
		prob_max =np.max(probs,axis=1)
		mask = np.argsort(prob_max)[::-1]

		conn.close()
		return jsonify({"%04d" % (i+1):{'Label':labels[mask][i],'Category':prob_arg_max[mask][i],'MaxProbability':prob_max[mask][i]} for i in range(len(mask))})

class TopSort_Predictions(Resource):

	def get(self,predictiondate):
		PredictionDay = dt.strptime(predictiondate,'%Y-%m-%d').date()

		conn = db_connect.connect()

		query = conn.execute("select label,probcat0,probcat1,probcat2,probcat3,probcat4,probcat5,probcat6,probcat7,probcat8,probcat9,probcat10,probcat11 from prediction3BT where predictionday = ?",(PredictionDay,))
		
		tmp = query.cursor.fetchall()
		labels= np.array([_tmp[0] for _tmp in tmp])
		probs = np.array([_tmp[1:] for _tmp in tmp])
		prob_arg_max = np.argmax(probs,axis=1)
		prob_max =np.max(probs,axis=1)
		mask = np.lexsort((prob_max,prob_arg_max))[::-1]

		conn.close()
		return jsonify({"%04d" % (i+1):{'Label':labels[mask][i],'Category':prob_arg_max[mask][i],'MaxProbability':prob_max[mask][i]} for i in range(len(mask))})

api.add_resource(StockInfo,'/stocks')
api.add_resource(Stock_Prediction_Day,'/stocks/<string:stock_label>/<string:predictiondate>')
api.add_resource(Stock_Prediction_Day_All,'/stocks/<string:stock_label>/predictions')
api.add_resource(Single_Stock_Info,'/stocks/<string:stock_label>')
api.add_resource(Top_Predictions,'/top/<string:predictiondate>')
api.add_resource(TopSort_Predictions,'/top_sort/<string:predictiondate>')

if __name__ == '__main__':
	#app.run(debug=True)
	app.run(host='0.0.0.0')