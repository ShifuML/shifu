package ml.shifu.shifu.core.dtrain;


import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.GuaguaMRUnitDriver;
import ml.shifu.guagua.unit.GuaguaUnitDriver;
import ml.shifu.shifu.core.dtrain.multitask.MTNNMaster;
import ml.shifu.shifu.core.dtrain.multitask.MTNNParams;
import ml.shifu.shifu.core.dtrain.multitask.MTNNWorker;
import org.testng.annotations.Test;

import java.sql.Driver;
import java.util.Properties;

/**
 * @author haillu
 */
public class MultiTaskNNTest {

    @Test
    public void testMultiTaskNN() {
        Properties props = new Properties();
        props.setProperty(GuaguaConstants.MASTER_COMPUTABLE_CLASS, MTNNMaster.class.getName());
        props.setProperty(GuaguaConstants.WORKER_COMPUTABLE_CLASS, MTNNWorker.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_ITERATION_COUNT, "10");
        props.setProperty(GuaguaConstants.GUAGUA_MASTER_RESULT_CLASS, MTNNParams.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_WORKER_RESULT_CLASS, MTNNParams.class.getName());
        props.setProperty(CommonConstants.MODELSET_SOURCE_TYPE, "LOCAL");
        String modelConfigJson = getClass().getResource("/model/MultiTaskNN/ModelConfig.json").toString();
        props.setProperty(CommonConstants.SHIFU_MODEL_CONFIG, modelConfigJson
                );
        props.setProperty(CommonConstants.SHIFU_COLUMN_CONFIG,
                getClass().getResource("/model/MultiTaskNN/ColumnConfig.json").toString());
        props.setProperty(GuaguaConstants.GUAGUA_INPUT_DIR,
                getClass().getResource("/data/part-00").toString());

        GuaguaUnitDriver<MTNNParams, MTNNParams> driver = new GuaguaMRUnitDriver<MTNNParams, MTNNParams>(props);
        driver.run();
    }
}
