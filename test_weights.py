from extract_coupleWeights import extract_coupleWeights
from py_utils.data_managment import get_files, load, save


if __name__ == "__main__":
    # job_data = load(r'C:\Users\aless\Desktop\gTec_EEGpipeline\data\models\me\me.20250922.1938.mi_lhrh.joblib')
    # for k in job_data.keys():
    #     if not k.startswith('__'):
    #         print(k, job_data[k].shape)
    #         print(job_data[k])
    extract_coupleWeights(gammaMI=0, gammaRest=0, doSave=True)