package ml.shifu.shifu.core.dtrain.dt;

import org.apache.log4j.BasicConfigurator;
import org.apache.log4j.Level;
import org.apache.log4j.LogManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeMethod;
import org.testng.annotations.Test;

import java.io.File;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Created by haifwu on 2016/12/16.
 */
public class DTEarlyStopDeciderTest {
    private static final Logger LOG = LoggerFactory.getLogger(DTEarlyStopDeciderTest.class);

    private List<Double> trainErrorList = new ArrayList<Double>();
    private List<Double> validationErrorList = new ArrayList<Double>();
    private final static String DATA_FILE_NAME = "dttest/data/trainError.dat";

    @BeforeMethod
    public void setUp() throws Exception {
        BasicConfigurator.configure();
        LogManager.getRootLogger().setLevel(Level.DEBUG);
        ClassLoader classLoader = getClass().getClassLoader();
        URL url = classLoader.getResource("trainError.dat");
        File file = new File(this.getClass().getClassLoader().getResource(DATA_FILE_NAME).getFile());

        Scanner scanner = new Scanner(file);
        while (scanner.hasNext()){
            String line = scanner.nextLine();
            String[] info = line.split("\\t");
            if(info.length != 2){
                LOG.error("Wrong format of line: " + line);
            }
            trainErrorList.add(Double.valueOf(info[0]));
            validationErrorList.add(Double.valueOf(info[1]));
        }
    }

    @Test
    public void testAdd() throws Exception {
        DTEarlyStopDecider dtEarlyStopDecider = new DTEarlyStopDecider(6);
        Assert.assertEquals(trainErrorList.size(), validationErrorList.size());

        int iteration = 0;
        while (iteration ++ < trainErrorList.size()){
            if(dtEarlyStopDecider.add(trainErrorList.get(iteration), validationErrorList.get(iteration))){
                LOG.info("Iteration " + iteration + " stop!");
                break;
            }
        }

        Assert.assertNotSame(iteration, trainErrorList.size());
    }

}