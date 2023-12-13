from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling

model = BayesianNetwork([('UV','BKL'),('UV','MEL'),('UV','BCC'),('UV','AKIEC'),('RAD','BCC'),('GEN','MEL'),('AGE','AKIEC'),('GEN','DF'),('AGE','MEL'),
                         ('AGE','VASC'),('AGE','BCC'),('AGE','BKL'),('TRA','VASC'),('TRA','DF'),('MEL','BLE'),('MEL','MOL'),('VASC','BLE'),('BCC','BLE'),('BCC','BUM'),
                         ('DF','SMO'),('DF','BUM'),('AKIEC','SCA'),('BKL','SCA')])

# Risk Factors CPDs
cpd_uv = TabularCPD(variable ='UV', variable_card = 2, values = [[0.5],[0.5]])
cpd_tra = TabularCPD(variable ='TRA', variable_card = 2, values = [[0.5],[0.5]])
cpd_gen = TabularCPD(variable ='GEN', variable_card = 2, values = [[0.0005],[0.9995]])
cpd_age = TabularCPD(variable ='AGE', variable_card = 3, values = [[0.3],[0.4],[0.3]])
cpd_rad = TabularCPD(variable ='RAD', variable_card = 2, values = [[0.5],[0.5]])

# Diseases CPDs
cpd_mel = TabularCPD(variable='MEL', variable_card = 2, values = [[0.00015,0.0003,0.000225,0.0001,0.00012,0.00024,0.000075,0.00015,0.0001125,0.00006,0.00012,0.00009],
                                                                  [0.99985,0.9997,0.999775,0.99988,0.99988,0.99976,0.999925,0.99985,0.9998875,0.99994,0.99988,0.99991]], 
                     evidence = ['UV','GEN','AGE'], evidence_card = [2,2,3])

cpd_df = TabularCPD(variable='DF', variable_card = 2, values = [[0.072,0.06,0.036,0.03],[0.928,0.94,0.964,0.97]], 
                     evidence = ['TRA','GEN'], evidence_card = [2,2])

cpd_bkl = TabularCPD(variable='BKL', variable_card = 2, values = [[0.6,0.78,0.9,0.3,0.39,0.45],[0.4,0.22,0.1,0.7,0.61,0.55]], 
                     evidence = ['UV','AGE'], evidence_card = [2,3])

cpd_akiec = TabularCPD(variable='AKIEC', variable_card = 2, values = [[0.088,0.1144,0.1584,0.044,0.0572,0.0792],[0.912,0.8856,0.8416,0.956,0.9428,0.9208]], 
                     evidence = ['UV','AGE'], evidence_card = [2,3])

cpd_bcc = TabularCPD(variable='BCC', variable_card = 2, values = [[0.006716,0.00292,0.003358,0.00146],[0.993284,0.99708,0.996642,0.99854]], 
                     evidence = ['UV','RAD'], evidence_card = [2,2])

cpd_vasc = TabularCPD(variable='VASC', variable_card = 2, values = [[0.0007,0.0011,0.0015,0.00035,0.00055,0.00075],[0.9993,0.9989,0.9985,0.99965,0.99945,0.99925]], 
                     evidence = ['TRA','AGE'], evidence_card = [2,3])


# Signs CPDs
cpd_sca = TabularCPD(variable='SCA', variable_card = 2, values = [[0.8,0.65,0.75,0.0001],[0.2,0.35,0.25,0.9999]], 
                     evidence = ['BKL','AKIEC'], evidence_card = [2,2])

cpd_smo = TabularCPD(variable='SMO', variable_card = 2, values = [[0.75,0.2],[0.25,0.8]], 
                     evidence = ['DF'], evidence_card = [2])

cpd_mol = TabularCPD(variable='MOL', variable_card = 2, values = [[0.9,0.003],[0.1,0.997]], 
                     evidence = ['MEL'], evidence_card = [2])

cpd_bum = TabularCPD(variable='BUM', variable_card = 2, values = [[0.98,0.98,0.40,0.01],[0.02,0.02,0.60,0.99]], 
                     evidence = ['DF','BCC'], evidence_card = [2,2])

cpd_ble = TabularCPD(variable='BLE', variable_card = 2, values = [[0.70,0.60,0.65,0.55,0.65,0.55,0.60,0.20],[0.30,0.40,0.35,0.45,0.35,0.45,0.40,0.80]], 
                     evidence = ['MEL','BCC','VASC'], evidence_card = [2,2,2])


model.add_cpds(cpd_uv,cpd_tra,cpd_gen,cpd_age,cpd_bkl,cpd_rad,cpd_df,cpd_mel,cpd_akiec,cpd_bcc,cpd_vasc,cpd_sca,cpd_smo,cpd_mol,cpd_bum,cpd_ble)
model.check_model()


# Exact inference (variable elimination)
evidence={"UV":1, "TRA":1, "GEN":0, "AGE":0, "MOL":0, "BUM":0, "BLE":0, "SCA":0, "SMO":1, "RAD":1}

infer = VariableElimination(model)
posterior_p = infer.query(["AKIEC"], evidence=evidence)
print(posterior_p)

# Approx inference (Rejection Sampling)
sampler = BayesianModelSampling(model)
samples = sampler.rejection_sample(evidence=evidence.items(),size=10000)
print(samples['AKIEC'].value_counts(normalize=True))

# Approx inference (Likelihood Weighting)
sampler = BayesianModelSampling(model)
samples = sampler.likelihood_weighted_sample(evidence=evidence.items(),size=10000)
print(samples['AKIEC'].value_counts(normalize=True))