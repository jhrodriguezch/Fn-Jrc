# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 19:02:08 2021

@author: ICJRC
"""
import numpy as np, sys, pandas as pd

class ModeloSocialZapatosa():
    # PRINCIPAL METHODS
    def __init__(self,
        n_especies = 12,
        n_metodos  = 3,
        dt         = 0.05,
        n_pasostiempo = 14):
        
        '''
        DEFAULT VALUES
        
        n_especies = Numero de especies
        n_metodos  = Numero de artes de pesca
        dt         = paso de tiempo
        n_pasostiempo = numero de pasos de tiempo

        '''
        # data for fix the data
        self.n_especies    = n_especies
        self.n_metodos     = n_metodos
        self.dt            = dt # ano / paso de tiempo
        self.n_pasostiempo = n_pasostiempo
        
        # initial running
        self.t = np.arange(0, n_pasostiempo + dt, dt)
        
        self.LoadParameters()
        self.LoadData()
        self.TestSize()
        
    def __call__(self,
        INIT_HaCult = 20_514,
        INIT_IngA   = 657370.6858244,
        INIT_P      = 149_697,
        INIT_Pp     = 9996.09819207691,
        INIT_Pa     = 98295.0216317322,
        INIT_Pg     = 59118.053513589,
        INIT_G      = 200000.000002635,
        INIT_HaPas  = 218_043.95,
        INIT_Pflot  = 31_596):
        
        '''# MAIN OF THE PROGRAM
        Obj: CORRER MODELO DE ZAPATOZA

        # VARIABLES
        
        INIT_HaCult = Ha de cultivo inicial
        INIT_G      = Cantidad inicial de ganado
        INIT_HaPas  = Ha de pastos inicial
        INIT_P      = Habitante totales inicial
        INIT_Pp     = Poblacion pesquera inicial
        INIT_Pa     = Poblacion agricola inicial
        INIT_Pg     = poblacion ganadera inicial
        INIT_Pflot  = Poblacion flotante inicial
        '''
        
        # SAVE VALUES
        lts_t       = [self.t[0]]
        lts_Ha_Cult = [INIT_HaCult]
        lts_IngA    = [INIT_IngA]
        lts_p       = [INIT_P]
        lts_Pp      = [INIT_Pp]
        lts_Pa      = [INIT_Pa]
        lts_Pg      = [INIT_Pg]
        lts_G       = [INIT_G]
        lts_Hp      = [INIT_HaPas]
        lts_pflot   = [INIT_Pflot]
        
        lts_IngPes  = [np.nan]
        lts_A       = [np.nan]
        lts_IngGan  = [np.nan]
        lts_Pf      = [np.nan]
        
        lts_pes     = [np.nan]
        
        self.__LoadInitialData(INIT_HaCult, INIT_IngA, INIT_P, INIT_Pp, INIT_Pa, INIT_Pg, INIT_G, INIT_HaPas, INIT_Pflot)
        
        t = self.t[0]
        
        for num, tstep in enumerate(self.t[1:]):
            
            dt = tstep - t
            
            # INITIAL
            # Pesca FLOWS
            Disponibilidad = self.dt * self.__disponibilidad()
            pes            = self.dt * self.__pes(Disponibilidad, lts_Pp[-1])
            IngPes         = self.dt * self.__ingPes(pes)
            
            # Agricola FLOWS
            Ha_Cult        = self.__Ha_Cult(INIT_HaCult, dt)
            InvAgr         = self.__invAgr(INIT_IngA)
            A              = self.__Alim(InvAgr, Ha_Cult)
            IngAgr         = self.__IngAgr(A)
            
            # Ganaderia FLOWS
            HaPas, SHaPas  = self.__HaPas(INIT_HaPas, dt)
            
            # # STOKE
            G_i            = self.__G(HaPas, INIT_G, dt)
            IngGan         = self.__IngG(G_i)
            
            # # Poblacion FLOWS
            dpflot_dt, Pflot = self.__pflot(IngGan, IngAgr, IngPes, INIT_Pflot, INIT_Pg, INIT_Pa, INIT_Pp, dt)
            P, dP_dt         = self.__population(INIT_P, dpflot_dt, dt)
            
            Pp, dPp_dt     = self.__pobActiv('-p', IngGan, INIT_Pg, IngAgr, INIT_Pa, IngPes, INIT_Pp, dP_dt, dt)
            Pa, dPa_dt     = self.__pobActiv('-a', IngGan, INIT_Pg, IngAgr, INIT_Pa, IngPes, INIT_Pp, dP_dt, dt)
            Pg, dPg_dt     = self.__pobActiv('-g', IngGan, INIT_Pg, IngAgr, INIT_Pa, IngPes, INIT_Pp, dP_dt, dt)
            
            # # REINITIAL VALUES
            INIT_HaCult = Ha_Cult
            INIT_IngA   = IngAgr  # [COP/ano]
            
            # # Valores iniciales de la poblacion
            INIT_P      = P                # [Hab]
            INIT_Pp     = Pp               # [Hab]
            INIT_Pa     = Pa               # [Hab]
            INIT_Pg     = Pg               # [Hab]
            INIT_HaPas  = SHaPas           # [Ha]
            INIT_Pflot  = Pflot            # [Hab]
            
            # # Valores iniciales de la Actividades economicas
            INIT_G      = G_i               # [cab]
            
            # SAVE VALUES
            lts_t.append(tstep)
            lts_Ha_Cult.append(INIT_HaCult)
            lts_IngA.append(INIT_IngA)
            lts_p.append(INIT_P)
            lts_pflot.append(INIT_Pflot)
            lts_Pp.append(INIT_Pp)
            lts_Pa.append(INIT_Pa)
            lts_Pg.append(INIT_Pg)
            lts_G.append(INIT_G)
            
            lts_IngPes.append(IngPes)
            lts_A.append(A)
            lts_IngGan.append(IngGan)
            lts_Pf.append(dpflot_dt)
            
            lts_pes.append(pes)
            lts_Hp.append(INIT_HaPas)
            
            t = tstep
            
            
        self.RESULT = pd.DataFrame()
        self.RESULT['tiempo'] = self.t
        self.RESULT['Ha_Culti'] = lts_Ha_Cult
        self.RESULT['Ing_Agri'] = lts_IngA
        
        self.RESULT['Poblacio'] = lts_p
        self.RESULT['Pob_Pesq'] = lts_Pp
        self.RESULT['Pob_Agri'] = lts_Pa
        self.RESULT['Pob_Gana'] = lts_Pg
        self.RESULT['Pob_Flotante']=lts_pflot
        
        self.RESULT['C_Ganado'] = lts_G
        
        self.RESULT['Ing_Pesq'] = lts_IngPes
        self.RESULT['Alimento'] = lts_A
        self.RESULT['Ing_Gand'] = lts_IngGan
        self.RESULT['Pob_flot_dpf_dt'] = lts_Pf
        
        self.RESULT['Pesca']    = lts_pes
        self.RESULT['Ha_pasto'] = lts_Hp
        
        self.RESULT = self.RESULT.set_index('tiempo').copy()
        
    # USEFULL METHODS
    
    # HIDEN MODEL METHODS STOKES
    def __G(self, HaPas, Init, dt):
        k1 = self.PARA_theta_16 * HaPas - self.PARA_theta_3 * Init
        k2 = self.PARA_theta_16 * HaPas - self.PARA_theta_3 * (Init + (1 / 2) * k1)
        k3 = self.PARA_theta_16 * HaPas - self.PARA_theta_3 * (Init + (1 / 2) * k2)
        k4 = self.PARA_theta_16 * HaPas - self.PARA_theta_3 * (Init + (1 / 1) * k3)

        dG_dt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        G_i = self.__correcData(Init + dt * dG_dt)
        return(G_i)
    
    # HIDEN MODEL METHODS FLOWS
    @staticmethod
    def __correcData(num):
        return(0 if num <= 0 else num)
    
    def __Ha_Cult(self, Init, dt):
        k1 = Init * self.DATA_rate_HaCultivos * self.PARA_theta_17
        k2 = (Init + (1 / 2) * k1) * self.DATA_rate_HaCultivos * self.PARA_theta_17
        k3 = (Init + (1 / 2) * k2) * self.DATA_rate_HaCultivos * self.PARA_theta_17
        k4 = (Init + (1 / 1) * k3) * self.DATA_rate_HaCultivos * self.PARA_theta_17
        dHa_Cult_dt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        Ha_Cult = self.__correcData(self.INIT_HaCult + dt * dHa_Cult_dt)

        # Ha_Cult = self.INIT_HaCult + self.INIT_HaCult * self.DATA_rate_HaCultivos * self.PARA_theta_17\
        #     if self.INIT_HaCult + self.INIT_HaCult * self.DATA_rate_HaCultivos * self.PARA_theta_17 >= 0 else 0 
        return(Ha_Cult)
    
    def __pes(self, Disponibilidad, INIT_Pp):
        # Pesca [ton / ano]
        # Flow
        # Supocicion : Se pesca una cantidad independiente de la disponibilidad
        # de peces, esta solo se usa para comparar
        pes = np.minimum(self.PARA_theta_12 * INIT_Pp * self.PARA_theta_20 * self.DATA_ProdMet, Disponibilidad)
        # print ('Se pesca toda la disponibilidad') if (pes == Disponibilidad).all() else None
        return(pes)
    
    def __disponibilidad(self):
        # Disponibilidad para la pesca [ton / ano]
        # Flow
        # Supocicion : Las sanciones afectan independientemente de la especie o
        # la metodologia usada.
        Disponibilidad = self.DATA_DispBiol -  self.PARA_theta_13 * self.DATA_ActIlic * np.ones([self.n_metodos,self.n_especies])
        Disponibilidad[Disponibilidad <= 0] = 0.0
        return(Disponibilidad)
    
    def __HaPas(self, Init, dt):
        # Area disponible para la ganaderia [Ha / ano]
        # Flow
        # Supocicion : El cambio del area de la cienaga solo contribuye o afecta
        # al area para llevar a acabo la ganaderia
        
        k1 =  Init                 * self.DATA_rate_HaPasUS_Ha - self.PARA_theta_15 * self.DATA_HaCienagas * self.DATA_rate_HaCienagas
        k2 = (Init + (1 / 2) * k1) * self.DATA_rate_HaPasUS_Ha - self.PARA_theta_15 * self.DATA_HaCienagas * self.DATA_rate_HaCienagas
        k3 = (Init + (1 / 2) * k2) * self.DATA_rate_HaPasUS_Ha - self.PARA_theta_15 * self.DATA_HaCienagas * self.DATA_rate_HaCienagas
        k4 = (Init + (1 / 1) * k3) * self.DATA_rate_HaPasUS_Ha - self.PARA_theta_15 * self.DATA_HaCienagas * self.DATA_rate_HaCienagas
        
        dHaPas_dt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        SHaPas = self.__correcData(Init + dt * dHaPas_dt)
        return(dHaPas_dt, SHaPas)
    
    def __invAgr(self, IngA):
        # Inversion realizada en cultivos agricolas [cop / ano]
        # Flow
        InvAgr = self.PARA_theta_14 * IngA
        return(InvAgr)
    
    def __Alim(self, InvAgr, Ha_Cult):
        # Cantidad de alimento cosachado [ton / ano]
        # Flow
        A = self.PARA_theta_10 * Ha_Cult + self.PARA_theta_11 * InvAgr
        return(A)
    
    def __IngAgr(self, A):
        # Ingresos agricolas generados en un ano [cop / ano]
        # Flow
        IngAgr = self.PARA_theta_4 * (1 - self.DATA_PorcAutA) * A
        return(IngAgr)
    
    def __ingPes(self, pes):
        # Ingresos pesqueros generados en un ano [cop / ano]
        # Flow
        IngPes = np.sum((1 - self.PARA_theta_6) * np.sum(self.PARA_theta_5 * pes * (1 - self.DATA_PorcAutP).T, axis=1))
        return(IngPes)
    
    def __IngG(self, G_i):
        IngG = self.PARA_theta_2 * (1 - self.DATA_PorcAutG) * self.PARA_theta_3 * G_i
        return(IngG)
    
    def __pflot(self, IngG, IngA, IngP, Init, InitPg, InitPa, InitPp, dt):
        # [Hab / anio]
        # FLOW
        IngGPP = IngG / int(InitPg) if InitPg > 0 | ~ np.isnan(InitPg) else 0
        IbgAPP = IngA / int(InitPa) if InitPa > 0 | ~ np.isnan(InitPg) else 0
        IngPPP = IngP / int(InitPp) if InitPp > 0 | ~ np.isnan(InitPg) else 0
        
        pflot_in  = self.PARA_theta_1 * ((1/3.) * (IngGPP + IbgAPP + IngPPP))
        pflot_out = self.PARA_theta18 * ((1/3.) * (IngGPP + IbgAPP + IngPPP))
        
        dpflot_dt = dt * (pflot_in - pflot_out)
        
        # STOCK
        Pflot = self.__correcData(Init + dpflot_dt)
        
        return(dpflot_dt, Pflot)
    
    def __population(self, Init, dpf_dt, dt):
        k1 = (Init                  / 1000.) * self.PARA_theta_19 * (self.DATA_nata - self.DATA_mort) + dpf_dt
        k2 = ((Init + (1 / 2) * k1) / 1000.) * self.PARA_theta_19 * (self.DATA_nata - self.DATA_mort) + dpf_dt
        k3 = ((Init + (1 / 2) * k2) / 1000.) * self.PARA_theta_19 * (self.DATA_nata - self.DATA_mort) + dpf_dt
        k4 = ((Init + (1 / 1) * k3) / 1000.) * self.PARA_theta_19 * (self.DATA_nata - self.DATA_mort) + dpf_dt
        
        dP_dt = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        
        P = self.__correcData(Init + dt * dP_dt)
        return(P, dP_dt)
    
    def __pobActiv(self, idAct, IngGan, Pg, 
                                IngAgr, Pa,
                                IngPes, Pp,
                                dP_dt, dt):
        
        if idAct == '-p':
            theta_7 = self.PARA_theta_7_p
            theta_8 = self.PARA_theta_8_p
            IngX     = IngPes
            PorAutoX = self.DATA_PorcAutP.mean()
            Px_ini   = Pp
        elif idAct == '-a':
            theta_7 = self.PARA_theta_7_a
            theta_8 = self.PARA_theta_8_a
            IngX     = IngAgr
            PorAutoX = self.DATA_PorcAutA
            Px_ini   = Pa
        elif idAct == '-g':
            theta_7 = self.PARA_theta_7_g
            theta_8 = self.PARA_theta_8_g
            IngX     = IngGan
            PorAutoX = self.DATA_PorcAutG
            Px_ini   = Pg
        else:
            sys.exit('Non idAct correct')
        
        
        dPx_dt = dP_dt * (theta_7 * 1 / 10_000 * (IngX /(IngGan + IngAgr + IngPes)) + theta_8 \
                        * (PorAutoX /(self.DATA_PorcAutA + self.DATA_PorcAutG + \
                                         self.DATA_PorcAutP.mean())))
            
        Px = self.__correcData(Px_ini + dt * dPx_dt)
        
        return(Px, dPx_dt)
    
    # HIDEN FIX METHODS
    def TestSize(self):
        print('Check VarTest 1') if list(self.PARA_theta_5.shape)   == [self.n_metodos, self.n_especies] else sys.exit('Error en las dimensiones del parametro theta 5 ')
        print('Check VarTest 2') if list(self.PARA_theta_6.shape)   == [self.n_metodos]                  else sys.exit('Error en las dimensiones del parametro theta 6')
        print('Check VarTest 3') if list(self.PARA_theta_12.shape)  == [self.n_metodos, 1]               else sys.exit('Error en las dimensiones del parametro theta 12')
        print('Check VarTest 4') if list(self.DATA_DispBiol.shape)  == [self.n_metodos, self.n_especies] else sys.exit('Error en las dimensiones de la variable DispBiol')
        print('Check VarTest 5') if list(self.DATA_ProdMet.shape)   == [self.n_metodos, self.n_especies] else sys.exit('Error en las dimensiones de la variable ProdMet')
        print('Check VarTest 6') if list(self.DATA_PorcAutP.shape)  == [self.n_especies, 1]              else sys.exit('Error en las dimensiones de la variable PorcAutP')
        
    
    def LoadParameters(self,
        PARA_theta_1 = 2.32958772687529,
        PARA_theta_2 = 518_867.48,
        PARA_theta_3 = 0.053,
        PARA_theta_4 = 2_177_136.22,
        PARA_theta_5 = [1_763_696,
                        0,
                        0,
                        2_882_964.52,
                        0,
                        5_379_272.80,
                        0,
                        0,
                        5_048_579.80,
                        0,
                        0,
                        0],
        PARA_theta_6 = [0.1, 0.1, 0.1],
        PARA_theta_7_p = 4.18932678516848,
        PARA_theta_7_a = 0.150481295033305,
        PARA_theta_7_g = 2.59444606928667,
        PARA_theta_8_p = 9.99173295616787,
        PARA_theta_8_a = 0.103266865037611,
        PARA_theta_8_g = 4.61860246244377,
        PARA_theta_10 = 6.38,
        PARA_theta_11 = 2.5775985054404476e-05, 
        PARA_theta_12 =[[0.20808335798562466],[0.10798605825727428],[0.15628649391145602]],
        PARA_theta_13 = 1.6917563998052831,
        PARA_theta_14 = 6.817035073167162e-06,
        PARA_theta_15 = 0.144281324141783,
        PARA_theta_16 = -0.380984432117318,
        PARA_theta_17 = 4.16810371571129,
        PARA_theta18  = 2.312137453334,
        PARA_theta_19 = 1.15421708263406,
        PARA_theta_20 = 2946.669882718011):
        
        '''
        OBJ : CARGAR LOS PARAMETROS DEL MODELO:
            
        PARA_theta_1 [Hab * Hab / COP]
            
        PARA_theta_2 [COP / Cabeza de ganado]
            
        * PARA_theta_3 [Cab ganado vendida / Cab ganado total * ano]
            
        PARA_theta_4 [COP de venta / ton de alimento]
        
        PARA_theta_5 [COP / ton de pescado]
        
        PARA_theta_6 [COP invertido para realizar un arte de pesca / COP ganado
                      por el arte de pesca]
        
        PARA_theta_7_x [Nuevos Hab debido a los ingresos que se dedican a la 
                        actividad x / Habitantes totales]
        
        PARA_theta_8_x [Nuevos Hab debido a la autoalimentacion que se dedican
                        a la actividad x / Habitantes totales]
        
        * PARA_theta_10 [ton de alimento / ha cultivo * ano]
        
        PARA_theta_11 [ton alimento / COP invertido en la cosecha]
        
        PARA_theta_12 [Hab que usa la metodologia i / Hab dedicado a la pesca]
        
        PARA_theta_13 [ton pescado no desembarcado / Sancion]
        
        PARA_theta_14 [COP invertido en el cultivo / COP ganado por la cosecha]
        
        PARA_theta_15 [Ha pasto / Ha cienaga]
        
        PARA_theta_16 [CAB / Ha pasto]
        
        PARA_theta_17 = Corrector parameter
        
        PARA_theta18  = Porcentaje de la poblacion flotante que se va del sector de estudio
        
        PARA_theta_19 = Correccion de la tasa de natalidad y mortandad
        
        PARA_theta_20 = Correccion de la produtividad de la pesca

        DEFAULT VALUES ---
        
        PARA_theta_1 = 0.01697088609014997,
        PARA_theta_2 = 518_867.48,
        PARA_theta_3 = 0.053,
        PARA_theta_4 = 2_177_136.22,
        PARA_theta_5 = [1_763_696,
                        0,
                        0,
                        2_882_964.52,
                        0,
                        5_379_272.80,
                        0,
                        0,
                        5_048_579.80,
                        0,
                        0,
                        0],
        PARA_theta_6 = [0.1, 0.1, 0.1],
        PARA_theta_7_p = 0.06353198737277023,
        PARA_theta_7_a = 0.15498507367474063,
        PARA_theta_7_g = 0.300061166287696,
        PARA_theta_8_p = 0.14225814250347601,
        PARA_theta_8_a = 0.04061477450613044,
        PARA_theta_8_g = 0.5979137270359538,
        PARA_theta_10 = 6.38,
        PARA_theta_11 = 2.5775985054404476e-05, 
        PARA_theta_12 =[[0.20808335798562466],[0.10798605825727428],[0.15628649391145602]],
        PARA_theta_13 = 1.6917563998052831,
        PARA_theta_14 = 6.817035073167162e-06,
        PARA_theta_15 = 0.07686135601699137,
        PARA_theta_16 = 0.5475852897256095,
        PARA_theta_17 = 0.5446196391537492,
        PARA_theta18  = 0.01695261877625509,
        PARA_theta_19 = 0.03171039088194487,
        PARA_theta_20 = 2946.669882718011

        '''
        self.PARA_theta_1 = PARA_theta_1
        
        self.PARA_theta_2 = PARA_theta_2
        
        self.PARA_theta_3 = self.dt * PARA_theta_3
        
        self.PARA_theta_4 = PARA_theta_4
        
        self.PARA_theta_5 = np.array([PARA_theta_5, ] * self.n_metodos)
        
        self.PARA_theta_6 = np.array(PARA_theta_6)
        
        self.PARA_theta_7_p = PARA_theta_7_p
        self.PARA_theta_7_a = PARA_theta_7_a
        self.PARA_theta_7_g = PARA_theta_7_g
        
        self.PARA_theta_8_p = PARA_theta_8_p
        self.PARA_theta_8_a = PARA_theta_8_a
        self.PARA_theta_8_g = PARA_theta_8_g
        
        self.PARA_theta_10 = self.dt * PARA_theta_10
        
        self.PARA_theta_11 = PARA_theta_11
        
        self.PARA_theta_12 = np.array(PARA_theta_12)
        self.PARA_theta_13 = PARA_theta_13
        
        self.PARA_theta_14 = PARA_theta_14
        
        self.PARA_theta_15 = PARA_theta_15
        
        self.PARA_theta_16 = PARA_theta_16
        
        self.PARA_theta_17 = PARA_theta_17
        
        self.PARA_theta18 = PARA_theta18
        
        self.PARA_theta_19 = PARA_theta_19
        
        self.PARA_theta_20 = PARA_theta_20
        
    def LoadData(self,
        DATA_DispBiol          = [[781.731675,220.870814,220.870814,261.250605,378.703282,217.763466,25.7037613,756.55411,257.24738,1843.69138,186.366998,1338.64745],
                                 [0,0,0,0,0,0,0,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,0,0,0,0]],
        DATA_ActIlic           = 1,
        
        DATA_HaPasUS_Ha        = 218_043.95,
        DATA_rate_HaPasUS_Ha   = -0.10066388,
        
        DATA_HaCienagas        = 49_240.391,
        DATA_rate_HaCienagas   = 0.4587,
        
        DATA_ProdMet  = [[0.2416304348	,0.0357065217	,0.5424456522	,0.6743478261	,0.5078804348	,3.1576630435	,2.1968478261	,0.0004891304	,0.532826087	,0.0065217391	,0.0661956522	,0.0407608696],
                         [3.8660869565	,0.5713043478	,8.6791304348	,10.7895652174	,8.1260869565	,50.5226086957	,35.1495652174	,0.007826087	,8.5252173913	,0.1043478261	,1.0591304348	,0.652173913],
                         [7.0072826087	,1.0354891304	,15.730923913	,19.5560869565	,14.7285326087	,91.5722282609	,63.7085869565	,0.0141847826	,15.4519565217	,0.1891304348	,1.919673913	,1.1820652174]],
        DATA_PorcAutA = 0.420897305576836,
        DATA_PorcAutG = 0.444415414264785,
        DATA_PorcAutP = [[0.125],
                        [0.000],
                        [0.000],
                        [0.125],
                        [0.000],
                        [0.125],
                        [0.000],
                        [0.000],
                        [0.15125],
                        [0.000],
                        [0.000],
                        [0.000]],
        DATA_nata = 14.58,
        DATA_mort = 4.56,
        
        DATA_rate_HaCultivos = 0.7253):
        
        ''' CARGAR DATOS DE LA CIENAGA:
        
        PARAMETROS ---
            
        * DATA_DispBiol : Disponibilidad biologica [TON/ANO]
                    |Mojarra amarilla  |-----------------|Chango                |Comelon                |Mojarra Tilapia      |Nicuro        |----------------------|-----------------------|Bocachico             |Bagre rayado                  |Gara Gara               |Arenca
        			|Caquetaia kraussii|Curimata mivartii|Cyphocharax magdalenae|Megaleporinus muyscorum|Oreochromis niloticus|Pimelodus yuma|Plagioscion magdalenae|Potamotrygon magdalenae|Prochilodus magdalenae|Pseudoplatystoma magdaleniatum|Trachelyopterus insignis|Triportheus magdalenae 
        ATARRAYA	|781.731675 	   |220.870814 	     |515.933241 	        |261.250605 	        |378.703282 	      |217.763466 	 |25.7037613 	        |756.55411 	            |257.24738 	           |1843.69138 	                  |186.366998 	           |1338.64745 
        CHINCHORRO	|     0            |     0           |     0                |     0                 |     0               |     0        |     0                |     0                 |     0                |     0                        |     0                  |     0                      
        OTROS    	|     0            |     0           |     0                |     0                 |     0               |     0        |     0                |     0                 |     0                |     0                        |     0                  |     0                          
        
        * DATA_ActIlic : Actividad ilicita (Sanciones / Ano)
        
        DATA_HaPasUS_Ha: Area de pastos del sector a analizar [Ha pasto]
        
        * DATA_rate_HaPasUS_Ha : Tasa de cambio del area de pasto 
        [Ha pasto cambian/ha pasto * ano]
        
        * DATA_HaCienagas: Area de cienagas del sector a analizar [Ha agua]
        
        * DATA_rate_HaCienagas : Tasa de cambio del area de cienagas 
        [Ha agua cambian/ha agua * ano]
        
        * DATA_ProdMet : Produccion de la pesca por cada metodologia y para cada 
        especie [Ton / Hab * ano]
        
        DATA_PorcAutA : Porcentaje de autoconsumo de la produccion agricola 
        [ton / ton]
        
        DATA_PorcAutG : Porcentaje de autoconsumo de la produccion ganadera 
        [cab / cab]
        
        DATA_PorcAutP : Porcentaje de autoconsumo de la produccion pesquera 
        [Ton / Ton]
        
        * DATA_nata : Tasa de natalidad [1000 hab / ano]
        
        * DATA_mort : Tasa de mortalidad [1000 hab / ano]
        
        DEFAULT VALUES ---
        
        DATA_DispBiol          = [[781.731675,220.870814,220.870814,261.250605,378.703282,217.763466,25.7037613,756.55411,257.24738,1843.69138,186.366998,1338.64745],
                                 [0,0,0,0,0,0,0,0,0,0,0,0],
                                 [0,0,0,0,0,0,0,0,0,0,0,0]],
        DATA_ActIlic           = 1,
        
        DATA_HaPasUS_Ha        = 218_043.95,
        DATA_rate_HaPasUS_Ha   = -0.05066388,
        
        DATA_HaCienagas        = 49_240.391,
        DATA_rate_HaCienagas   = 0.4587,
        
        DATA_ProdMet  = [[0.2416304348	,0.0357065217	,0.5424456522	,0.6743478261	,0.5078804348	,3.1576630435	,2.1968478261	,0.0004891304	,0.532826087	,0.0065217391	,0.0661956522	,0.0407608696],
                         [3.8660869565	,0.5713043478	,8.6791304348	,10.7895652174	,8.1260869565	,50.5226086957	,35.1495652174	,0.007826087	,8.5252173913	,0.1043478261	,1.0591304348	,0.652173913],
                         [7.0072826087	,1.0354891304	,15.730923913	,19.5560869565	,14.7285326087	,91.5722282609	,63.7085869565	,0.0141847826	,15.4519565217	,0.1891304348	,1.919673913	,1.1820652174]],
        DATA_PorcAutA = 0.37620352128186463,
        DATA_PorcAutG = 0.08585747030620694,
        DATA_PorcAutP = [[0.125],
                        [0.000],
                        [0.000],
                        [0.125],
                        [0.000],
                        [0.125],
                        [0.000],
                        [0.000],
                        [0.15125],
                        [0.000],
                        [0.000],
                        [0.000]],
        DATA_nata = 14.58,
        DATA_mort = 4.56,
        
        DATA_rate_HaCultivos = 0.7253
        
        '''
        
        self.DATA_DispBiol          = np.array(DATA_DispBiol)
        
        self.DATA_ActIlic           = DATA_ActIlic
        
        self.DATA_HaPasUS_Ha        = DATA_HaPasUS_Ha
        self.DATA_rate_HaPasUS_Ha   = DATA_rate_HaPasUS_Ha
        
        self.DATA_HaCienagas        = DATA_HaCienagas
        self.DATA_rate_HaCienagas   = DATA_rate_HaCienagas
        
        self.DATA_ProdMet           = np.array(DATA_ProdMet)
        
        self.DATA_PorcAutA          = DATA_PorcAutA
        self.DATA_PorcAutG          = DATA_PorcAutG
        
        self.DATA_PorcAutP          = np.array(DATA_PorcAutP)
        
        self.DATA_nata              = DATA_nata
        
        self.DATA_mort              = DATA_mort
        
        self.DATA_rate_HaCultivos   = DATA_rate_HaCultivos
    
    def __LoadInitialData(self, INIT_HaCult, INIT_P, INIT_HaPas, INIT_Pflot,
                          INIT_G, INIT_IngA , INIT_Pp, INIT_Pa,
                          INIT_Pg):
        '''
        DEFAULT VALUES
        INIT_IngA = 839208451.7240292, 
        INIT_Pp = 7158.45484151997,
        INIT_Pa = 95416.55349237971,
        INIT_Pg = 31000.22427019107
        '''
        # Valores iniciales Agricolas
        self.INIT_HaCult = INIT_HaCult # [Ha]
        self.INIT_HaPas  = INIT_HaPas # [Ha]
        self.INIT_IngA   = INIT_IngA  # [COP/ano]
        
        # Valores iniciales de la poblacion
        self.INIT_P      = INIT_P    # [Hab]
        self.INIT_Pp     = INIT_Pp   # [Hab]
        self.INIT_Pa     = INIT_Pa   # [Hab]
        self.INIT_Pg     = INIT_Pg   # [Hab]
        self.INIT_Pflot  = INIT_Pflot
        
        # Valores iniciales de la Actividades economicas
        self.INIT_G      = INIT_G    # [cab]

# a = ModeloSocialZapatosa()
# a()
# b = a.RESULT.copy()
# import matplotlib.pyplot as plt
# for ii in b.columns:
#     if not(ii == 'Pesca'):
#         b[ii].plot()
#         plt.legend()
#         plt.show()