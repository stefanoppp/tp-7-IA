from neurona_final import Neurona_final
from neuronas_ocultas import Neurona_oculta
import matplotlib.pyplot as plt
import random
import cv2
class Back_Propagation():
    
    def __init__(self,entradas,salidas):
        self.entradas=entradas
        self.salidas=salidas
    
    def main(self):
        
        cant_neuronas=int(input("Digite cantidad de neuronas: "))
        iteraciones=int(input("Digite cantidad de iteraciones: "))
        
        neuronas=[]
        # instanciamos neuronas
        for i in range(cant_neuronas):
            pesos_neuronales=[]
            for j in range(len(self.entradas[0])):
                peso_random=random.uniform(-0.01,0.01)
                pesos_neuronales.append(peso_random)
            n=Neurona_oculta(pesos_neuronales)
            neuronas.append(n)
            pesos_finales=[]
        #  instanciamos neurona final   
        for j in range(len(neuronas)+1):
            peso_random=random.random()
            pesos_finales.append(peso_random)
        nf=Neurona_final(pesos_finales)
        errores=[]
        # comienza la itereacion
        for iteracion in range(iteraciones):
            for i in range(len(self.entradas)):
                salidas_ocultas=[]
                for neurona in neuronas:
                    salida_oculta=neurona.obtener_salida(self.entradas[i])
                    salidas_ocultas.append(salida_oculta)
                # agregamos bias final
                salidas_ocultas.append(1)
                # recalculamos pesos finales
                salida_red=nf.obtener_salida(salidas_ocultas)
                error_red=nf.obtener_error(self.salidas[i],salida_red)
                errores.append(error_red)
                delta_final=nf.obtener_delta_final(salida_red,error_red)
                variaciones=[]
                for salida in salidas_ocultas:
                    variacion=nf.variacion_pesos(salida,delta_final)
                    variaciones.append(variacion)
                
                nf.calcular_nuevos_pesos(variaciones)
                # recalculamos pesos ocultos
                variaciones=[]
                for neurona in neuronas:
                    salida=neurona.obtener_salida(self.entradas[i])
                    delta_oculto=neurona.obtener_delta_oculto(salida,delta_final)
                    for entrada in self.entradas[i]:
                        variacion=neurona.variacion_pesos(entrada,delta_oculto)
                        variaciones.append(variacion)
                    neurona.calcular_nuevos_pesos(variaciones)
            print("Iteracion ",iteracion+1)
            print(error_red)
            
        while True:
            foto=input("Ingrese la foto que desea probar: ")
            image = cv2.imread(foto)
            pixeles=[]
            for alto in image:
                for ancho in alto:
                    pixeles.append(ancho[0])
            pixeles.append(1)
            pixeles_fotos.append(pixeles)
            
            salidas=[]
            for neurona in neuronas:
                salida=neurona.obtener_salida(pixeles)
                salidas.append(salida)
            salidas.append(1)
            salida_final=nf.obtener_salida(salidas)
            print(round(salida_final))
            print("\n")
        # -------------------GRAFICA DE ERRORES------------------        
        # array=[]
        # for i in range(len(self.entradas)):
        #     array.append([])
        
        # j=0
        # for i in range(len(errores)):
        #     array[j].append(errores[i])
        #     j+=1
        #     if j==len(array):
        #         j=0
        # for element in array:
        #     plt.plot(element)
        # plt.show()



fotos=['1A57190.jpg',
       '1B57190.jpg',
       '2A57190.jpg',
       '2B57190.jpg',
       '3A57190.jpg',
       '3B57190.jpg',
       '4A57190.jpg',
       '4B57190.jpg',
        '5A57190.jpg',
       '5B57190.jpg',
       ]

salidas=[1,0,1,0,1,0,1,0,1,0]

pixeles_fotos=[]

for foto in fotos:
    image = cv2.imread(foto)
    auxiliar=[]
    for alto in image:
        for ancho in alto:
            auxiliar.append(ancho[0])
    auxiliar.append(1)
    pixeles_fotos.append(auxiliar)
back_2=Back_Propagation(pixeles_fotos,salidas) 
back_2.main()   