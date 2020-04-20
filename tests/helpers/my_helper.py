from datetime import datetime


class MyHelper():

    @classmethod
    def days_ago(cls, d):
        return (datetime.now().date() - d).days
