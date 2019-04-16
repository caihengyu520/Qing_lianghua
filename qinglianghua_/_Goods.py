#产品类
class Goods:  # 产品类
    def __init__(self):
        self.count = 0

    def add(self, num=1):
        self.count += num

    def sub(self):
        if self.count >= 0:
            self.count =0

    def empty(self):
        return self.count <= 0

    def notempty(self):
        return self.count>0