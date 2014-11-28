package ml.shifu.shifu.varsel;

import java.io.IOException;
import java.util.Properties;

import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.mapreduce.GuaguaMRUnitDriver;
import ml.shifu.guagua.unit.GuaguaUnitDriver;
import ml.shifu.shifu.core.dvarsel.VarSelMaster;
import ml.shifu.shifu.core.dvarsel.VarSelMasterResult;
import ml.shifu.shifu.core.dvarsel.VarSelWorker;
import ml.shifu.shifu.core.dvarsel.VarSelWorkerResult;

import org.testng.annotations.Test;

public class VotedVarSelTest {

	//@Test
    public void testLrApp() throws IOException {
        Properties props = new Properties();
        
        props.setProperty(GuaguaConstants.MASTER_COMPUTABLE_CLASS, VarSelWorker.class.getName());
        props.setProperty(GuaguaConstants.WORKER_COMPUTABLE_CLASS, VarSelMaster.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_ITERATION_COUNT, "20");
        props.setProperty(GuaguaConstants.GUAGUA_MASTER_RESULT_CLASS, VarSelMasterResult.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_WORKER_RESULT_CLASS, VarSelWorkerResult.class.getName());

        props.setProperty(GuaguaConstants.GUAGUA_INPUT_DIR, getClass().getResource("/example/cancer-judgement/DataStore/DataSet1/part-00").toString());

        GuaguaUnitDriver<VarSelMasterResult, VarSelWorkerResult> driver = new GuaguaMRUnitDriver<VarSelMasterResult, VarSelWorkerResult>(props);

        driver.run();


	}

}
