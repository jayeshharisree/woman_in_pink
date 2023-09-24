import os
import redis

# For Remote DB
db=redis.from_url("redis://WomanInPink:PF1tIEOnebmQDHTvmG5EXroxM8YVKu7T@redis-10056.c114.us-east-1-4.ec2.cloud.redislabs.com:10056")

# For Local DB
# db=redis.Redis(host='localhost', port=6379, password='')


def adduser(email):
	if db.hsetnx("users","email",email):
		return True
	else:
		return False


def getusers():
	return db.hgetall("users")


def validateuser(email):
	user=db.hget("users", email)
	if(user is None ):
		return False
	else:
		user=user.replace('0','1')
		db.hset("users", email, user)
		return True


def deluser(email):
	status=db.hdel("users",email)
	print(email)
	print(status)
	if(status):
		return True
	else:
		return False