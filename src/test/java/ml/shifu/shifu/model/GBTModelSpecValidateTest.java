package ml.shifu.shifu.model;

import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.dtrain.dt.Node;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;
import org.junit.Assert;
import org.testng.annotations.Test;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.Map;

/**
 * Created by zhanhu on 5/23/17.
 */
public class GBTModelSpecValidateTest {

    @Test
    public void validateCAMGBT() throws IOException {
        Assert.assertTrue(validate("src/test/resources/dttest/model/model_cam.gbt"));
    }

    // @Test
    public void validateShifuGBTModel() throws IOException {
        Assert.assertTrue(validate("~/Downloads/model0.gbt"));
    }

    // @Test
    public void validateShifuGBT2Model() throws IOException {
        Assert.assertTrue(validate("src/test/resources/dttest/model0.gbt"));
    }

    private boolean validate(String modelLoc) throws IOException {
        IndependentTreeModel treeModel = IndependentTreeModel.loadFromStream(new FileInputStream(modelLoc));
        boolean status = true;
        for(TreeNode tree: treeModel.getTrees().get(0)) {
            Node node = tree.getNode();
            status = status && validate(node, treeModel.getNumNameMapping());
        }
        return status;
    }

    private boolean validate(Node node, Map<Integer, String> numNameMapping) {
        if(node != null && !node.isRealLeaf()) {
            Integer splitColumnNum = node.getSplit().getColumnNum();
            if(!numNameMapping.containsKey(splitColumnNum)) {
                return false;
            } else {
                return validate(node.getLeft(), numNameMapping) && validate(node.getRight(), numNameMapping);
            }
        }
        return true;
    }
}
