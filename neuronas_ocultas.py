import math
import numpy as np
class Neurona_oculta:
    def __init__(self,pesos):
        self.pesos=pesos
        self.lr=0.85
        
    def obtener_salida(self,entradas):
        prod_escalar=np.dot(self.pesos,entradas)
        salida_real=self.sigmoidea(prod_escalar)
        return salida_real
    
    def obtener_delta_oculto(self,salida_obtenida,delta_final):
        delta_oculto=salida_obtenida*(1-salida_obtenida)*delta_final
        return delta_oculto
    
    def variacion_pesos(self,entrada,delta_oculto):
        variacion=entrada*self.lr*delta_oculto 
        return variacion   
    
    def calcular_nuevos_pesos(self,variaciones):
        for i in range(len(self.pesos)):
            self.pesos[i]=self.pesos[i]+variaciones[i]

    def sigmoidea(self,prod_escalar):
        sig = 1 / (1 + math.exp(-prod_escalar))
        return sig