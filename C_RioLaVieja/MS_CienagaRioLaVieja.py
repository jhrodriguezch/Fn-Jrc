# -*- coding: utf-8 -*-
"""
@author: mrsao
"""
import numpy as np, pandas as pd
import sys

# MODELO SOCIAL
class MS_CienagaRioLaVieja():
    def __init__(self, init_year, end_year):
        self.init_year = init_year
        self.end_year = end_year
        self.dictionary()
        
    def __call__(self,
                Pas_ini_ha,
                Cul_tra_ini_ha, Cul_per_ini_ha, Cul_het_ini_ha,
                Caf_ini_ha,
                Gan_ini_cb,
                Pob_ini_ru,
                Pob_ini_cb):
        
        self.res = self.model(Pas_ini_ha, 
                              Cul_tra_ini_ha, Cul_per_ini_ha, Cul_het_ini_ha,\
                              Caf_ini_ha, Gan_ini_cb, Pob_ini_ru, Pob_ini_cb)
        self.modelExtr()
    
    # RUN
    def model(self,
             Pas_ini_ha,
             Cul_tra_ini_ha, Cul_per_ini_ha, Cul_het_ini_ha,
             Caf_ini_ha,
             Gan_ini_cb,
             Pob_ini_ru,
             Pob_ini_cb):
        
        # Assig values
        
        Pc_cul  = self.dict['Habitantes requeridos por ha de cultivos']
        Pc_caf  = self.dict['Habitantes requeridos por ha de cultivo de cafe']
        Pg_cab  = self.dict['Habitantes requeridos por cabeza de ganado']
        tur_HaC = self.dict['Turistas atraidos por cada ha de cafe']
        P_tur   = self.dict['Habitantes requeridos por el sector turistico']
        Ps_P    = self.dict['Habitantes requeridos por el sector servicios']
        Pru_IDR = self.dict['Migracion rural atraida']
        Pcb_IDU = self.dict['Migracion cabecera atraida']
        
        T_CulTr   = self.dict['Tasa de cambio del cultivo - Transitorio']
        Tend_AreaCulTr = self.dict['Area tendencial del cultivo - Transitorio']
        T_CulPr   = self.dict['Tasa de cambio del cultivo - Permanente']
        Tend_AreaCulPr = self.dict['Area tendencial del cultivo - Permanente']
        T_CulHt   = self.dict['Tasa de cambio del cultivo - Heterogeneo']
        Tend_AreaCulHt = self.dict['Area tendencial del cultivo - Heterogeneo']
        
        T_cafe = self.dict['Tasa de cambio del cultivo de cafe']
        Tend_AreaCaf = self.dict['Area tendencial del cultivo de cafe']
        
        Soporte = self.dict['Ganado soportado por una ha de pasto']
        T_crec  = self.dict['Tasa de crecimiento natural del ganado bovino']
        T_sacrif = self.dict['Tasa de sacrificio del ganado bovino']
        Requerimie = self.dict['Ha de pasto requerido por el ganado']
        
        T_cnat_rul = self.dict['Tasa de crecimiento natural - Población rural']
        T_cnat_cab = self.dict['Tasa de crecimiento natural - Población cabecera']
        
        # Run model
        Cul_ini_ha = Cul_tra_ini_ha + Cul_per_ini_ha + Cul_het_ini_ha
        
        ano = [self.init_year]
        Pas_lts = [Pas_ini_ha]
        
        Ctr_lts = [Cul_tra_ini_ha]
        Cpr_lts = [Cul_per_ini_ha]
        Cht_lts = [Cul_het_ini_ha]
        
        Cul_lts = [Cul_ini_ha]
        Caf_lts = [Caf_ini_ha]
        Gan_lts = [Gan_ini_cb]
        
        Pru_lts = [Pob_ini_ru]
        Pca_lts = [Pob_ini_cb]
        
        tur_lts = [tur_HaC * Caf_ini_ha]
        PDES_lts   = [np.nan]
        
        for year in np.arange(self.init_year + 1 , self.end_year + 1):
            
            # JOIN ECO to POB
            
            Pcul = int(Pc_cul * Cul_ini_ha)
            Pcaf = int(Pc_caf * Caf_ini_ha)
            Pgan = int(Pg_cab * Gan_ini_cb)
            tur  = int(tur_HaC * Caf_ini_ha)
            Ptur = int(P_tur * tur)
            Pser = int(Ps_P * Pob_ini_cb)
            
            try:
                PDR = (Pcul + Pcaf + Pgan + Ptur) / Pob_ini_ru
            except:
                PDR = 0
            
            try:
                PDU = Pser / Pob_ini_cb
            except:
                PDU = 0
            
            try:
                PDES_lts.append((Pob_ini_ru + Pob_ini_cb - Pcul - Pcaf - Pgan - Ptur - Pser) / (Pob_ini_ru + Pob_ini_cb))
            except:
                PDES_lts.append(1)
                
            ME_ru = Pru_IDR * PDR
            ME_cb = Pcb_IDU * PDU
            
            # 4.1. Area de cultivo
            Cul_tra = self.RK4(init=Cul_tra_ini_ha, h=1, fs=[T_CulTr/4, Tend_AreaCulTr], fun=self.dAc_dt)
            Cul_tra = 0 if Cul_tra <= 0 else Cul_tra
            
            Cul_per = self.RK4(init=Cul_per_ini_ha, h=1, fs=[T_CulPr/4, Tend_AreaCulPr], fun=self.dAc_dt)
            Cul_per = 0 if Cul_per <= 0 else Cul_per
            
            Cul_het = self.RK4(init=Cul_het_ini_ha, h=1, fs=[T_CulHt/4, Tend_AreaCulHt], fun=self.dAc_dt)
            Cul_het = 0 if Cul_het <= 0 else Cul_het
            
            Cul = Cul_tra + Cul_per + Cul_het
            
            # 4.2. Area de cultivo de cafe
            Caf = self.RK4(init=Caf_ini_ha, h=1, fs=[T_cafe/4, Tend_AreaCaf], fun=self.dAcafe_dt)
            Caf = 0 if Caf <= 0 else Caf
            
            # JOIN A to G
            Soport = Pas_ini_ha * Soporte
            
            # 4.3. Ganaderia values
            G = self.RK4(Gan_ini_cb, h=1, fs=[T_crec/4, T_sacrif/4, Soport], fun=self.dG_dt)
            G = 0 if G <= 0 else int(G)
            
            # JOIN G to A
            T_pas = Requerimie * ((G - Gan_ini_cb)/G) if G !=0 else 0
            
            # 4.4 Area values
            A = self.RK4(Pas_ini_ha, h=1, fs=[T_pas / 4], fun=self.dA_dt)
            A = 0 if A <= 0 else A
            
            # 4.5. Poblacion Rural
            P_rul = self.RK4(init=Pob_ini_ru, h=1, fs=[T_cnat_rul/4., ME_ru], fun=self.dPob_dt)
            P_rul = 0 if P_rul <= 0 else int(P_rul)
            
            # 4.6. Poblacion cabecera
            P_cab = self.RK4(init=Pob_ini_cb, h=1, fs=[T_cnat_cab/4., ME_cb], fun=self.dPob_dt)
            P_cab = 0 if P_cab <= 0 else int(P_cab)
    
            # 5. Add result to list
            Pas_lts.append(A)
            Gan_lts.append(G)
            
            Ctr_lts.append(Cul_tra)
            Cpr_lts.append(Cul_per)
            Cht_lts.append(Cul_het)
            Cul_lts.append(Cul)
            
            Caf_lts.append(Caf)
            Pru_lts.append(P_rul)
            Pca_lts.append(P_cab)
            tur_lts.append(tur)
            
            ano.append(year)
            
            # 6. Actualizar valores
            
            Cul_tra_ini_ha = Cul_tra
            Cul_per_ini_ha = Cul_per
            Cul_het_ini_ha = Cul_het
            Cul_ini_ha = Cul_tra_ini_ha + Cul_per_ini_ha + Cul_het_ini_ha
            Caf_ini_ha = Caf
            Gan_ini_cb = G
            Pas_ini_ha = A
            
            Pob_ini_ru = P_rul
            Pob_ini_cb = P_cab
            
        # 7. ADD RESULTS
        res = pd.DataFrame()
        res['Ano'] = ano
        
        res['Area Cultivo - Transitorio'] = Ctr_lts
        res['Area Cultivo - Permanente']  = Cpr_lts
        res['Area Cultivo - Heterogeneo'] = Cht_lts
        res['Area Cultivo']    = Cul_lts
        
        res['Area Cult. cafe'] = Caf_lts
        res['Area Pasto']      = Pas_lts
        res['Ganado']          = Gan_lts
        
        res['Pob. rural']      = Pru_lts
        res['Pob. cabecera']   = Pca_lts
        res['Pob. Turistas']   = tur_lts
        
        res['Tasa de desempeleo'] = PDES_lts
        
        # 8. RESET INDEX
        res.index = res['Ano']
        res.drop('Ano', axis=1, inplace=True)
        
        return(res)
    
    def modelExtr(self):
        
        # VARIABLES
        T_sac   = self.dict['Tasa de sacrificio del ganado bovino']
        kg_cab  = self.dict['Peso en canal por cabeza de ganado']
        COP_kg  = self.dict['Precio por kilogramo de carne en canal']
        ton_haC = self.dict['Rendimiento agricola']
        cop_kgC = self.dict['Precio de venta de la produccion agricola']
        ton_haCafe = self.dict['Rendimiento del cultivo de cafe']
        cop_kgcafe = self.dict['Precio del kilogramo cafe']
        DotNetaHab = self.dict['Dotacion neta promedio']
        PorcPerd = self.dict['Porcentqaje de perdidad']
        DotNetaBov = self.dict['Consumo de agua ganadero']
        
        # ADD CHARACTERISTICS
        self.res['Poblacion total'] = self.res['Pob. rural'] + self.res['Pob. cabecera']
        
        self.res['Ganado sacrificado'] = self.res['Ganado'] * T_sac
        self.res['Capital ganadero']    = self.res['Ganado sacrificado'] * kg_cab * COP_kg
        
        self.res['Produccion agricola'] = self.res['Area Cultivo'] * ton_haC
        self.res['Capital agricola'] = self.res['Produccion agricola'] * 1000 * cop_kgC
        
        self.res['Produccion de cafe'] = self.res['Area Cult. cafe'] * ton_haCafe
        self.res['Capital cafetero '] = self.res['Produccion de cafe'] * 1000 * cop_kgcafe
        
        self.res['Demanda de agua - uso domestico'] = (self.res['Poblacion total'] + self.res['Pob. Turistas']) * DotNetaHab * (1/86_400.) * (1 + PorcPerd)
        self.res['Demanda de agua - uso pecuario'] = self.res['Ganado'] * DotNetaBov * (1/86_400.)
        
        self.res['Demanda de agua - uso agricola - Cultivo Transitorio'] = self.DemAguaAgri(self.res['Area Cultivo - Transitorio'], coef_type = 'trn')
        self.res['Demanda de agua - uso agricola - Cultivo Permanente']  = self.DemAguaAgri(self.res['Area Cultivo - Permanente'], coef_type = 'per')
        self.res['Demanda de agua - uso agricola - Cultivo Heterogeneo'] = self.DemAguaAgri(self.res['Area Cultivo - Heterogeneo'], coef_type = 'htr')
        
        self.res['Demanda de agua - uso agricola'] = self.res['Demanda de agua - uso agricola - Cultivo Transitorio'] +\
            self.res['Demanda de agua - uso agricola - Cultivo Permanente'] + self.res['Demanda de agua - uso agricola - Cultivo Heterogeneo']
        
        self.res['Demanda de agua - uso agricola - cafetero'] = self.DemAguaAgri(self.res['Area Cult. cafe'], coef_type = 'cafe!')
        
        self.res['Demanda total de agua'] = self.res['Demanda de agua - uso domestico'] + self.res['Demanda de agua - uso pecuario'] +\
                                            self.res['Demanda de agua - uso agricola'] + self.res['Demanda de agua - uso agricola - cafetero']
        
        return(None)
    
    # MODELOS
    @staticmethod
    def dAcafe_dt(Acafe_ini, fs):
        return(Acafe_ini * fs[0] * (1 - Acafe_ini / fs[1]))
    
    @staticmethod
    def dAc_dt(A_ini, fs):
        return(A_ini * fs[0] * (1 - A_ini / fs[1]))
    
    @staticmethod
    def dA_dt(A_ini, fs):
        return(A_ini * fs[0])
    
    @staticmethod
    def dG_dt(G_ini, fs):
        return(G_ini*(fs[0] - fs[1])*(1 - G_ini/fs[2]))
    
    @staticmethod
    def dPob_dt(Pob_ini, fs):
        return(Pob_ini * fs[0] + fs[1])
    
    # Otros
    def DemAguaAgri(self, area_lts, coef_type):
        
        # 1. extraer valores por defecto
        umbralescorrentia = 25.020896     
        prec = np.array(self.dict['Precipitacion media mensual multianual'])
        evap = np.array(self.dict['Evapotranspiracion media mensual multianual'])
        
        # 2. extraer valores dependientes del calculo
        if coef_type == 'per':
            coef = np.array(self.dict['Coeficiente mensual de cultivos - Permanente'])
        elif coef_type == 'trn':
            coef = np.array(self.dict['Coeficiente mensual de cultivos - Transitorio'])
        elif coef_type == 'htr':
            coef = np.array(self.dict['Coeficiente mensual de cultivos - Heterogeneo'])
        elif coef_type == 'cafe!':
            coef = np.array(self.dict['Coeficiente mensual de cultivos - Cafe'])
        else:
            sys.exit('coef_type invalid!. Valids = [per, trn, htr, cafe!]')
        
        area = np.array(list(area_lts))
        area = np.expand_dims(area, axis=0)
        
        # 3. Calculo de la demanda por area
        prec_neta = ((prec - umbralescorrentia) ** 2) / ((prec + 4 * umbralescorrentia))
        demanda = (evap * coef - prec_neta) * (10_000/(31*86400))
        demanda[prec_neta > evap] = 0
        demanda = np.expand_dims(demanda, axis=0)
        
        # 4. Calcular la demanda de agua
        DemandaCalc = area.T * demanda
        DemandaCalc = np.max(DemandaCalc, axis=1)
        return(list(DemandaCalc))
    
    @staticmethod
    def RK4(init, h, fs, fun):
        subStep = 4
        h_int = h / subStep
        for ii in list(range(subStep)):
            k1 = fun(init          , fs)
            k2 = fun(init + k1 / 2., fs)
            k3 = fun(init + k2 / 2., fs)
            k4 = fun(init + k3     , fs)
            m = h_int * (k1 + 2 * k2 + 2 * k3 + k4) / 6.
            init = init + m
        return(init)

    def dictionary(self):
        self.dict = {
            # PARAMETROS CALIBRADOS
            'Tasa de crecimiento natural - Población rural' :0.4523786507420768,
            'Tasa de crecimiento natural - Población cabecera' :0.046158964648240236,
            
            'Habitantes requeridos por cabeza de ganado' : 0.03170731707317073,
            'Habitantes requeridos por ha de cultivos': 0.2567813901893124,
            'Habitantes requeridos por ha de cultivo de cafe': 1.1717936636226582,
            'Habitantes requeridos por el sector servicios': 0.6164363774925519,
            'Habitantes requeridos por el sector turistico': 3.1387512963254247,
            
            'Migracion rural atraida': -4073.705495428765,
            'Migracion cabecera atraida': -3156.063317267045,
            
            'Tasa de cambio del cultivo - Transitorio': 0.450155114,
            'Area tendencial del cultivo - Transitorio': 4917.606257,
            
            'Tasa de cambio del cultivo - Permanente': 1,
            'Area tendencial del cultivo - Permanente': 21365.7061,
            
            'Tasa de cambio del cultivo - Heterogeneo': 0.049495313,
            'Area tendencial del cultivo - Heterogeneo': 8691.083895,
            
            'Tasa de cambio del cultivo': 0.029019536330867172,
            'Area tendencial del cultivo': 20_000.0,
            
            'Tasa de cambio del cultivo de cafe': 0.5,
            'Area tendencial del cultivo de cafe': 28213.886641980112,
            
            'Tasa de crecimiento natural del ganado bovino': 0.8513797086002215,
            'Tasa de sacrificio del ganado bovino': 0.45,
            'Ha de pasto requerido por el ganado': 1.122484651827749,
            'Ganado soportado por una ha de pasto': 2,

            'Turistas atraidos por cada ha de cafe': 2.01651863734517,
            
            # PARAMENTRO EXTRA
            'Peso en canal por cabeza de ganado' : 245.08,
            'Precio por kilogramo de carne en canal' : 10_500,
            
            'Rendimiento agricola': 8.6019, # ton/ha
            'Precio de venta de la produccion agricola': 650, #cop/kg
            
            'Rendimiento del cultivo de cafe': 6.1, #ton/ha
            'Precio del kilogramo cafe': 15_500, # cop/kg
            
            'Dotacion neta promedio':95,# L/Hab/dia
            'Porcentqaje de perdidad': 0.17, # %
            
            'Consumo de agua ganadero': 86.40,# l/dia/cabeza
            
            'Precipitacion media mensual multianual':[128.50,
                                                      100.70,
                                                      243.06,
                                                      241.23,
                                                      265.54,
                                                      113.43,
                                                      42.89,
                                                      30.59,
                                                      176.01,
                                                      374.25,
                                                      222.00,
                                                      234.90],
            
            'Evapotranspiracion media mensual multianual':[109.56,
                                                           98.96,
                                                           110.80,
                                                           106.90,
                                                           110.03,
                                                           106.21,
                                                           109.12,
                                                           109.83,
                                                           105.93,
                                                           108.80,
                                                           104.90,
                                                           108.60],
            
            'Coeficiente mensual de cultivos - Permanente':[1.00, 1.00, 1.50,\
                                                            1.50, 1.50, 1.10,\
                                                            1.00, 1.00, 1.50,\
                                                            1.50, 1.50, 1.10],
            'Coeficiente mensual de cultivos - Transitorio':[1.20, 1.10, 1.20,\
                                                             1.20, 1.20, 1.20,\
                                                             1.20, 1.10, 1.20,\
                                                             1.20, 1.20, 1.20],
            'Coeficiente mensual de cultivos - Heterogeneo':[1.20, 1.10, 1.20,\
                                                             1.20, 1.20, 1.20,\
                                                             1.20, 1.10, 1.20,\
                                                             1.20, 1.20, 1.20],
            'Coeficiente mensual de cultivos - Cafe':[0.90, 0.90, 0.95,\
                                                      0.95, 0.95, 0.95,\
                                                      0.90, 0.90, 0.95,\
                                                      0.95, 0.95, 0.95],
            }
        return(None)
    
    def example(self):
        self.init_year = 2014
        self.end_year = 2050

        Pas_ini_ha = 70_163.1
        Cul_tra_ini_ha = 3_295.490896
        Cul_per_ini_ha = 18_525.14429
        Cul_het_ini_ha = 38_156.24065
        Caf_ini_ha = 31_001.4
        Gan_ini_cb = 113_781
        
        Pob_ini_ru = 93_964
        Pob_ini_cb = 672_154
        
        self.res = self.model(Pas_ini_ha, 
                              Cul_tra_ini_ha, Cul_per_ini_ha, Cul_het_ini_ha,
                              Caf_ini_ha, Gan_ini_cb, Pob_ini_ru, Pob_ini_cb)
        self.modelExtr()