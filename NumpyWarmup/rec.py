###### Jared Adams 00865234   <<<<<<<<<<<

class Rectangle:
     
    # add the init fucntion here that creates and initalize the
    # attributes of an object of this class. The attributes are
    # named as "length" and "width". The init function takes two
    # arguments that are to assigned to the length and width

    # write the init function here

  def __init__(self, length, width):
    self.length = length
    self.width = width
    
    
    # add another function here called "modify" that takes two arguments that
    # are to be assigned to the two attributes of the object.

  def modify(self, length, width):
    self.length = length
    self.width = width
    
    
    # add annother function here called "area" that is to compute
    # and return the area of the rectangle.

  def area(self):
    return self.length*self.width

    # add annother function here called "perimeter" that is to compute
    # and return the perimeter of the rectangle. 
  def perimeter(self):
    perimeter = (self.length*2)+(self.width*2)
    return perimeter

    
