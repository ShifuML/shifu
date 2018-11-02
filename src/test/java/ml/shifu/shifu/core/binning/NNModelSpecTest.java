package ml.shifu.shifu.core.binning;

import ml.shifu.shifu.core.dtrain.nn.NNMaster;
import ml.shifu.shifu.core.dtrain.nn.NNStructureComparator;
import org.encog.ml.BasicML;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.persist.EncogPersistor;
import org.testng.Assert;
import org.testng.annotations.Test;
import org.testng.collections.CollectionUtils;

import java.io.File;
import java.util.*;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class NNModelSpecTest {

    @Test
    public void testModelTraverse() {
        BasicML basicML = BasicML.class.cast(
                EncogDirectoryPersistence.loadObject(new File("src/test/resources/model/model0.nn")));
        BasicNetwork basicNetwork = (BasicNetwork) basicML;
        FlatNetwork flatNetwork = basicNetwork.getFlat();

        BasicML extendedBasicML = BasicML.class.cast(
                EncogDirectoryPersistence.loadObject(new File("src/test/resources/model/model1.nn")));
        BasicNetwork extendedBasicNetwork = (BasicNetwork) extendedBasicML;
        FlatNetwork extendedFlatNetwork = extendedBasicNetwork.getFlat();

        for ( int layer = flatNetwork.getLayerIndex().length - 1; layer > 0; layer -- ) {
            int layerOutputCnt = flatNetwork.getLayerFeedCounts()[layer - 1];
            int layerInputCnt = flatNetwork.getLayerCounts()[layer];

            System.out.println("Weight index for layer " + (flatNetwork.getLayerIndex().length - layer));

            int extendedLayerInputCnt = extendedFlatNetwork.getLayerCounts()[layer];

            int indexPos = flatNetwork.getWeightIndex()[layer - 1];
            int extendedIndexPos = extendedFlatNetwork.getWeightIndex()[layer - 1];

            for ( int i = 0 ; i < layerOutputCnt; i ++ ) {
                for ( int j = 0; j < layerInputCnt; j ++ ) {
                    int weightIndex = indexPos + (i * layerInputCnt) + j;
                    int extendedWeightIndex = extendedIndexPos + (i * extendedLayerInputCnt) + j;
                    if ( j == layerInputCnt - 1) { // move bias to end
                        extendedWeightIndex = extendedIndexPos
                                + (i * extendedLayerInputCnt) + (extendedLayerInputCnt - 1);
                    }

                    System.out.println(weightIndex + " --> " + extendedWeightIndex);
                }
            }
        }
    }

    @Test
    public void testFitExistingModelIn() {
        BasicML basicML = BasicML.class.cast(
                EncogDirectoryPersistence.loadObject(new File("src/test/resources/model/model0.nn")));
        BasicNetwork basicNetwork = (BasicNetwork) basicML;
        FlatNetwork flatNetwork = basicNetwork.getFlat();

        NNMaster master = new NNMaster();
        Set<Integer> fixedWeightIndexSet = master.fitExistingModelIn(flatNetwork, flatNetwork, Arrays.asList(new Integer[]{6}));
        List<Integer> indexList = new ArrayList<Integer>(fixedWeightIndexSet);
        Collections.sort(indexList);
        Assert.assertEquals(indexList.size(), 31);

        fixedWeightIndexSet = master.fitExistingModelIn(flatNetwork, flatNetwork, Arrays.asList(new Integer[]{1}));
        indexList = new ArrayList<Integer>(fixedWeightIndexSet);
        Collections.sort(indexList);
        Assert.assertEquals(indexList.size(), 930);

        BasicML extendedBasicML = BasicML.class.cast(
                EncogDirectoryPersistence.loadObject(new File("src/test/resources/model/model1.nn")));
        BasicNetwork extendedBasicNetwork = (BasicNetwork) extendedBasicML;
        FlatNetwork extendedFlatNetwork = extendedBasicNetwork.getFlat();
        fixedWeightIndexSet = master.fitExistingModelIn(flatNetwork, extendedFlatNetwork, Arrays.asList(new Integer[]{1}));
        indexList = new ArrayList<Integer>(fixedWeightIndexSet);
        Collections.sort(indexList);
        Assert.assertEquals(indexList.size(), 930);

        fixedWeightIndexSet = master.fitExistingModelIn(flatNetwork, extendedFlatNetwork, Arrays.asList(new Integer[]{1}), false);
        indexList = new ArrayList<Integer>(fixedWeightIndexSet);
        Collections.sort(indexList);
        Assert.assertEquals(indexList.size(), 900);
    }

    @Test
    public void testModelStructureCompare() {
        BasicML basicML = BasicML.class.cast(
                EncogDirectoryPersistence.loadObject(new File("src/test/resources/model/model0.nn")));
        BasicNetwork basicNetwork = (BasicNetwork) basicML;
        FlatNetwork flatNetwork = basicNetwork.getFlat();

        BasicML extendedBasicML = BasicML.class.cast(
                EncogDirectoryPersistence.loadObject(new File("src/test/resources/model/model1.nn")));
        BasicNetwork extendedBasicNetwork = (BasicNetwork) extendedBasicML;
        FlatNetwork extendedFlatNetwork = extendedBasicNetwork.getFlat();

        Assert.assertEquals(new NNStructureComparator().compare(flatNetwork, extendedFlatNetwork), -1);
        Assert.assertEquals(new NNStructureComparator().compare(flatNetwork, flatNetwork), 0);
        Assert.assertEquals(new NNStructureComparator().compare(extendedFlatNetwork, flatNetwork), 1);

        BasicML diffBasicML = BasicML.class.cast(
                EncogDirectoryPersistence.loadObject(new File("src/test/resources/model/model2.nn")));
        BasicNetwork diffBasicNetwork = (BasicNetwork) diffBasicML;
        FlatNetwork diffFlatNetwork = diffBasicNetwork.getFlat();
        Assert.assertEquals(new NNStructureComparator().compare(flatNetwork, diffFlatNetwork), -1);
        Assert.assertEquals(new NNStructureComparator().compare(diffFlatNetwork, flatNetwork), -1);
        Assert.assertEquals(new NNStructureComparator().compare(extendedFlatNetwork, diffFlatNetwork), 1);
        Assert.assertEquals(new NNStructureComparator().compare(diffFlatNetwork, extendedFlatNetwork), -1);

    }
}
