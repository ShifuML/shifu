package ml.shifu.shifu.core.dtrain;

import ml.shifu.shifu.combo.CsvFile;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.util.Constants;
import org.junit.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Created by zhanhu on 5/31/17.
 */
public class IndependentTreeModelTest {

    @Test
    public void testSplit() {
        Assert.assertEquals("aa", StringUtils.split("aa@^bb@cc", Constants.CATEGORICAL_GROUP_VAL_DELIMITER)[0]);
        Assert.assertEquals("bb@cc", StringUtils.split("aa@^bb@cc", Constants.CATEGORICAL_GROUP_VAL_DELIMITER)[1]);
        Assert.assertEquals("aa@bb@cc", StringUtils.split("aa@bb@cc", Constants.CATEGORICAL_GROUP_VAL_DELIMITER)[0]);
        Assert.assertEquals("", StringUtils.split("aa@^bb@^", Constants.CATEGORICAL_GROUP_VAL_DELIMITER)[2]);
        Assert.assertEquals("@", StringUtils.split("aa@^bb@^@", Constants.CATEGORICAL_GROUP_VAL_DELIMITER)[2]);
        Assert.assertEquals("@", StringUtils.split("@^bb@^@", Constants.CATEGORICAL_GROUP_VAL_DELIMITER)[2]);
    }

    //@Test
    public void testGBTTree() throws IOException {
        IndependentTreeModel treeModel = IndependentTreeModel.loadFromStream(
                IndependentTreeModelTest.class.getResourceAsStream("/example/encode/model0.gbt"));
        System.out.println(treeModel.getTrees().get(0).size());
    }

    //@Test
    public void testGBTTreeEncode() throws IOException {
        IndependentTreeModel treeModel = IndependentTreeModel.loadFromStream(
                IndependentTreeModelTest.class.getResourceAsStream("/example/encode/model0.gbt"));
        CsvFile csvFile = new CsvFile("src/test/resources/example/encode/sample.data.10", "\u0007");
        for (Map<String, String> rawData : csvFile) {
            Map<String, Object> input = new HashMap<String, Object>();
            input.putAll(rawData);
            List<String> instanceCodes = treeModel.encode(5, input);
            System.out.println(instanceCodes);
        }
    }
}
