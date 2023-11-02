from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD


model = BayesianNetwork([('UV','SUN'),('UV','MEL'),('MEC','DF'),('MEC','WOU'),('GEN','DF'),('GEN','AKIEK'),('RAD','BCC'),('AGE','BCC'),
                         ('SUN','COL'),('SUN','ITC'),('MEL','COL'),('MEL','MOL'),('MEL','ITC'),('DF','COL'),('DF','BUM'),('DF','ITC'),
                         ('WOU','ITC'),('WOU','BLE'),('AKIEK','ITC'),('AKIEK','BLE'),('AKIEK','SCA'),('BCC','BUM'),('BCC','SCA')])

cpd_uv = TabularCPD(variable ='UV', variable_card = 2, values = [[0.5],[0.5]])
cpd_mec = TabularCPD(variable ='MEC', variable_card = 2, values = [[0.5],[0.5]])
cpd_gen = TabularCPD(variable ='GEN', variable_card = 2, values = [[0.0005],[0.9995]])
cpd_rad = TabularCPD(variable ='RAD', variable_card = 2, values = [[0.0001],[0.9999]])
cpd_age = TabularCPD(variable ='AGE', variable_card = 2, values = [[0.5],[0.5]])

cpd_sun = TabularCPD(variable='SUN', variable_card = 2, values = [[0.95, 0],[0.05,1]], 
                     evidence = ['UV'], evidence_card = [2])

cpd_mel = TabularCPD(variable='MEL', variable_card = 2, values = [[0.0001, 0],[0.9999,1]], 
                     evidence = ['UV'], evidence_card = [2])

cpd_df = TabularCPD(variable='DF', variable_card = 2, values = [[0.008, 0.0005, 0.001,0],[0.992,0.9995,0.999,1]], 
                     evidence = ['MEC','GEN'], evidence_card = [2,2])

cpd_wou = TabularCPD(variable='WOU', variable_card = 2, values = [[1, 0],[0,1]], 
                     evidence = ['MEC'], evidence_card = [2])

cpd_akiek = TabularCPD(variable='AKIEK', variable_card = 2, values = [[0.003, 0],[0.997,1]], 
                     evidence = ['GEN'], evidence_card = [2])

cpd_BCC = TabularCPD(variable='BCC', variable_card = 2, values = [[0.3,0.15,0.05,0.0001],[0.7,0.85,0.95,0.9999]], 
                     evidence = ['RAD','AGE'], evidence_card = [2,2])


model.check_model()