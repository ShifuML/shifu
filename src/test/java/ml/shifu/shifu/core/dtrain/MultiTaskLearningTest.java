package ml.shifu.shifu.core.dtrain;


import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.GuaguaMRUnitDriver;
import ml.shifu.guagua.unit.GuaguaUnitDriver;
import ml.shifu.shifu.core.dtrain.mtl.MTLMaster;
import ml.shifu.shifu.core.dtrain.mtl.MTLParams;
import ml.shifu.shifu.core.dtrain.mtl.MTLWorker;
import org.testng.annotations.Test;

import java.util.Properties;

/**
 * @author haillu
 */
public class MultiTaskLearningTest {

    @Test
    public void testMultiTaskNN() {
        Properties props = new Properties();
        props.setProperty(GuaguaConstants.MASTER_COMPUTABLE_CLASS, MTLMaster.class.getName());
        props.setProperty(GuaguaConstants.WORKER_COMPUTABLE_CLASS, MTLWorker.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_ITERATION_COUNT, "10");
        props.setProperty(GuaguaConstants.GUAGUA_MASTER_RESULT_CLASS, MTLParams.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_WORKER_RESULT_CLASS, MTLParams.class.getName());
        props.setProperty(CommonConstants.MODELSET_SOURCE_TYPE, "LOCAL");
        String modelConfigJson = getClass().getResource("/model/MultiTaskNN/ModelConfig.json").toString();
        props.setProperty(CommonConstants.SHIFU_MODEL_CONFIG, modelConfigJson
                );
//        props.setProperty(CommonConstants.SHIFU_COLUMN_CONFIG,
//                getClass().getResource("/model/MultiTaskNN/ColumnConfig.json").toString());
        props.setProperty(GuaguaConstants.GUAGUA_INPUT_DIR,
                getClass().getResource("data/part-m-00000-mtl-afterNormalized").toString());

        GuaguaUnitDriver<MTLParams, MTLParams> driver = new GuaguaMRUnitDriver<MTLParams, MTLParams>(props);
        driver.run();
    }
}
