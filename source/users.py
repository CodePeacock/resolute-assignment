class Users:
    '''Class to store users'''
    def __init__(self):
        self.users = {}

    def add_user(self, user):
        self.users[user["name"]] = user

    def delete_user(self, name):
        del self.users[name]

    def get_users(self):
        return self.users

    def update_user(self, name, email):
        self.users[name]["email"] = email
