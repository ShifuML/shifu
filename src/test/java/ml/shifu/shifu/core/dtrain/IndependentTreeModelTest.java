package ml.shifu.shifu.core.dtrain;

import ml.shifu.shifu.util.Constants;

import org.junit.Assert;
import org.testng.annotations.Test;

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
}
