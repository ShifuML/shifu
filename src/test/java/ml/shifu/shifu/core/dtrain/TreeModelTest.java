package ml.shifu.shifu.core.dtrain;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.List;

import org.testng.annotations.BeforeClass;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.util.CommonUtils;

public class TreeModelTest {

    private TreeModel model;
    
    @BeforeClass
    public void setUp() throws IOException {
        String modelPath = "src/test/resources/example/wdbc/wdbcModelSetLocal/models/model0.gbt";
        FileInputStream fi = new FileInputStream(modelPath);
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList("src/test/resources/example/wdbc/wdbcModelSetLocal/ColumnConfig.json", SourceType.LOCAL);
        model = TreeModel.loadFromStream(fi, columnConfigList);
    }
    
    public void FeatureImportancesTest(){
        model.getFeatureImportances();
    }
    
    public static void main(String[] args) throws IOException{
        TreeModelTest test = new TreeModelTest();
        test.setUp();
        test.FeatureImportancesTest();
    }
}
