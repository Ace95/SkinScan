from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = BayesianNetwork([('UV','BKL'),('UV','NV'),('UV','MEL'),('UV','BCC'),('UV','AKIEC'),('GEN','MEL'),('GEN','BKL'),('GEN','AKIEC'),('GEN','DF'),('AGE','MEL'),
                         ('AGE','VASC'),('AGE','BCC'),('AGE','NV'),('TRA','VASC'),('TRA','DF'),('MEL','BLE'),('MEL','MOL'),('VASC','BLE'),('BCC','BLE'),('BCC','BUM'),
                         ('DF','SMO'),('DF','BUM'),('AKIEC','SCA'),('BKL','SCA'),('NV','MOL')])

# Risk Factors CPDs
cpd_uv = TabularCPD(variable ='UV', variable_card = 2, values = [[0.5],[0.5]])
cpd_tra = TabularCPD(variable ='TRA', variable_card = 2, values = [[0.5],[0.5]])
cpd_gen = TabularCPD(variable ='GEN', variable_card = 2, values = [[0.0005],[0.9995]])
cpd_age = TabularCPD(variable ='AGE', variable_card = 2, values = [[0.5],[0.5]])

# Diseases CPDs
cpd_mel = TabularCPD(variable='MEL', variable_card = 2, values = [[0.0001, 0,0.0001, 0,0.0001, 0,0.0001, 0],[0.9999,1,0.9999,1,0.9999,1,0.9999,1]], 
                     evidence = ['UV','GEN','AGE'], evidence_card = [2,2,2])

cpd_df = TabularCPD(variable='DF', variable_card = 2, values = [[0.008, 0.0005, 0.001,0],[0.992,0.9995,0.999,1]], 
                     evidence = ['TRA','GEN'], evidence_card = [2,2])

cpd_bkl = TabularCPD(variable='BKL', variable_card = 2, values = [[0.008, 0.0005, 0.001,0],[0.992,0.9995,0.999,1]], 
                     evidence = ['UV','GEN'], evidence_card = [2,2])

cpd_akiec = TabularCPD(variable='AKIEC', variable_card = 2, values = [[0.008, 0.0005, 0.001,0],[0.992,0.9995,0.999,1]], 
                     evidence = ['UV','GEN'], evidence_card = [2,2])

cpd_bcc = TabularCPD(variable='BCC', variable_card = 2, values = [[0.3,0.15,0.05,0.0001],[0.7,0.85,0.95,0.9999]], 
                     evidence = ['UV','AGE'], evidence_card = [2,2])

cpd_vasc = TabularCPD(variable='VASC', variable_card = 2, values = [[0.3,0.15,0.05,0.0001],[0.7,0.85,0.95,0.9999]], 
                     evidence = ['TRA','AGE'], evidence_card = [2,2])

cpd_nv = TabularCPD(variable='NV', variable_card = 2, values = [[0.3,0.15,0.05,0.0001],[0.7,0.85,0.95,0.9999]], 
                     evidence = ['UV','AGE'], evidence_card = [2,2])

# Symptoms CPDs
cpd_sca = TabularCPD(variable='SCA', variable_card = 2, values = [[0.3,0.15,0.05,0.0001],[0.7,0.85,0.95,0.9999]], 
                     evidence = ['BKL','AKIEC'], evidence_card = [2,2])

cpd_smo = TabularCPD(variable='SMO', variable_card = 2, values = [[0.3,0.15],[0.7,0.85]], 
                     evidence = ['DF'], evidence_card = [2])

cpd_mol = TabularCPD(variable='MOL', variable_card = 2, values = [[0.3,0.15,0.05,0.0001],[0.7,0.85,0.95,0.9999]], 
                     evidence = ['NV','MEL'], evidence_card = [2,2])

cpd_bum = TabularCPD(variable='BUM', variable_card = 2, values = [[0.3,0.15,0.05,0.0001],[0.7,0.85,0.95,0.9999]], 
                     evidence = ['DF','BCC'], evidence_card = [2,2])

cpd_ble = TabularCPD(variable='BLE', variable_card = 2, values = [[0.3,0.15,0.05,0.0001,0.3,0.15,0.05,0.0001],[0.7,0.85,0.95,0.9999,0.7,0.85,0.95,0.9999]], 
                     evidence = ['MEL','BCC','VASC'], evidence_card = [2,2,2])


model.add_cpds(cpd_uv,cpd_tra,cpd_gen,cpd_age,cpd_bkl,cpd_nv,cpd_df,cpd_mel,cpd_akiec,cpd_bcc,cpd_vasc,cpd_sca,cpd_smo,cpd_mol,cpd_bum,cpd_ble)
model.check_model()


# Exact inference (variable elimination)
infer = VariableElimination(model)
posterior_p = infer.query(["AKIEC"], evidence={"UV":1, "TRA":1, "GEN":0, "AGE":0, "MOL":0, "BUM":0, "BLE":0, "SCA":0, "SMO":1})
print(posterior_p)