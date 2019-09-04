package ml.shifu.shifu.core.dtrain;


import ml.shifu.guagua.GuaguaConstants;
import ml.shifu.guagua.hadoop.GuaguaMRUnitDriver;
import ml.shifu.guagua.unit.GuaguaUnitDriver;
import ml.shifu.shifu.core.dtrain.wdl.WDLMaster;
import ml.shifu.shifu.core.dtrain.wdl.WDLParams;
import ml.shifu.shifu.core.dtrain.wdl.WDLWorker;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

import java.util.Properties;

/**
 * Test run Wide And Deep Model in Local.
 *
 * @author Wu Devin (haifwu@paypal.com)
 */
public class WideAndDeepTest {
    private static final Logger LOG = LoggerFactory.getLogger(WideAndDeepTest.class);

    @Test
    public void testWideAndDeep() {
        Properties props = new Properties();
        props.setProperty(GuaguaConstants.MASTER_COMPUTABLE_CLASS, WDLMaster.class.getName());
        props.setProperty(GuaguaConstants.WORKER_COMPUTABLE_CLASS, WDLWorker.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_ITERATION_COUNT, "50");
        props.setProperty(GuaguaConstants.GUAGUA_MASTER_RESULT_CLASS, WDLParams.class.getName());
        props.setProperty(GuaguaConstants.GUAGUA_WORKER_RESULT_CLASS, WDLParams.class.getName());
        props.setProperty(CommonConstants.MODELSET_SOURCE_TYPE, "LOCAL");
        props.setProperty(CommonConstants.SHIFU_MODEL_CONFIG,
                getClass().getResource("/model/WideAndDeep/ModelConfig.json").toString());
        props.setProperty(CommonConstants.SHIFU_COLUMN_CONFIG,
                getClass().getResource("/model/WideAndDeep/ColumnConfig.json").toString());

        props.setProperty(GuaguaConstants.GUAGUA_INPUT_DIR, getClass().getResource("/data/part-m-00000-2").toString());

        GuaguaUnitDriver<WDLParams, WDLParams> driver = new GuaguaMRUnitDriver<>(props);

        driver.run();
    }
}
