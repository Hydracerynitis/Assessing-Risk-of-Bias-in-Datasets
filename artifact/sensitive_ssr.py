#Sensitive Subspace, in our development it was temporarily named as Sensitive Group
class SensitiveSubspace():
    def __init__(self,init_shapes=None):
        if init_shapes!=None:
            self.shapes=init_shapes
        else:
            self.shapes=[]

    def select(self,df):
        selector=~df.any(axis=1)
        for s in self.shapes:
            selector= selector | s.select(df)
        return selector
    
    def add(self,shape):
        if not isinstance(shape,SensitiveShape):
            return
        self.shapes.append(shape)
    
    def __str__(self) -> str:
        return "[{shape}]".format(shape="\n".join([str(s) for s in self.shapes]))
    def __repr__(self) -> str:
        return "[{shape}]".format(shape="\n".join([str(s) for s in self.shapes]))
    
    def get_json_repr(self,name):
        json_repr={}
        for i in range(len(self.shapes)):
            json_repr.update(self.shapes[i].get_json_repr(i))
        return {f"group {name}":json_repr}

    def parse_json(data):
        result=SensitiveSubspace()
        for v in data.values():
            result.add(SensitiveShape.parse_json(v))
        return result

class SensitiveShape():
    def __init__(self,init_rules=None):
        if init_rules!=None:
            self.rules=init_rules
        else:
            self.rules=[]
    
    def select(self,df):
        selector=df.any(axis=1)
        for sr in self.rules:
            selector= selector & sr.select(df)
        return selector
    
    def add(self,rule):
        if self.find(rule)!=None:
            return
        self.rules.append(rule)

    def get_rule(self,index):
        if isinstance(index,int):
            if index <0 or index>=len(self.rules):
                raise IndexError(f"Input index in range of 0 to {len(self.rules)}")
            else:
                return self.rules[index]
        if isinstance(index,str):
            for r in self.rules:
                if r.feature==index:
                    return r
            raise ValueError("Wrong Feature")

    def join(self,rule):
        r=self.find(rule)
        if r==None or not r.joinable(rule):
            return
        new_r=r.join(rule)
        new_shape=SensitiveShape([new_r]+[i for i in self.rules if i.feature!=r.feature])
        return new_shape

    def find(self,other):
        if isinstance(other,SensitiveRule):
            for r in self.rules:
                if r.feature==other.feature:
                    return r    
            return
        else:
            return
        
    def get_json_repr(self,name):
        json_repr={}
        for i in range(len(self.rules)):
            json_repr.update(self.rules[i].get_json_repr(i)) 
        return {f"shape {name}":json_repr}
    
    def parse_json(data):
        result=SensitiveShape()
        for v in data.values():
            result.add(SensitiveRule.parse_json(v))
        return result

    def __str__(self) -> str:
        return str(self.rules)
    def __repr__(self) -> str:
        return str(self.rules)

class SensitiveRule():
    def __init__(self,feature,bottom,top ) -> None:
        if top==bottom:
            raise ValueError("The upperbound of the rule can not be the same as the lower bound of the rule")
        self.top=top
        self.bottom=bottom
        self.feature=feature
    
    def joinable(self,other):
        return isinstance(other,SensitiveRule) and self.feature==other.feature and (self.top==other.bottom or self.bottom==other.top)

    def join(self, other):
        if self.joinable(other):
            return SensitiveRule(self.feature,min(self.bottom,other.bottom),max(self.top,other.top))
        
    def get_json_repr(self,name):
        json_repr={}
        json_repr["feature"]=self.feature
        json_repr["bottom"]=self.bottom
        json_repr["top"]=self.top
        return {f"rule {name}":json_repr}
    
    def parse_json(data):
        return SensitiveRule(data["feature"],data["bottom"],data["top"])

    def select(self,df):
        return  (df[self.feature] >= self.bottom) & (df[self.feature] <self.top)
    def __str__(self) -> str:
        return f"{self.feature}: {round(self.bottom,4)}~{round(self.top,4)}"
    def __repr__(self) -> str:
        return f"{self.feature}: {round(self.bottom,4)}~{round(self.top,4)}"
    
    def __eq__(self, other):
        if type(other) is type(self):
            return (self.feature,self.bottom,self.top)==(other.feature,other.bottom,other.top)
        else:
            return False

    def __hash__(self):
        return hash((self.feature,self.bottom,self.top))

class InsensitiveSubspace:
    def __init__(self,df,sensitive_groups) -> None:
        selector=df.any(axis=1)
        for sg in sensitive_groups:
            selector= selector & ~sg.select(df)
        self.selector=selector

    def select(self,df):
        return self.selector
    
class CatagoricalSubspace:
    def __init__(self,predicates) -> None:
        self.predicates=predicates

    def select(self,df):
        selector=df.any(axis=1)
        for pre in self.predicates:
            selector= selector & pre(df)
        return selector