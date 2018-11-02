package ml.shifu.shifu.util;

import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;
import java.util.zip.ZipOutputStream;

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

public class IndependentTreeModelUtils {

    public static final Logger logger = LoggerFactory.getLogger(IndependentTreeModelUtils.class);

    public static final String MODEL_CONF  = "model.ini";
    public static final String MODEL_TREES = "trees";


    public boolean convertBinaryToZipSpec(File treeModelFile, File outputZipFile) {
        FileInputStream treeModelInputStream = null;
        ZipOutputStream zipOutputStream = null;

        try {
            treeModelInputStream = new FileInputStream(treeModelFile);
            IndependentTreeModel treeModel = IndependentTreeModel.loadFromStream(treeModelInputStream);
            List<List<TreeNode>> trees = treeModel.getTrees();
            treeModel.setTrees(null);
            if (CollectionUtils.isEmpty(trees)) {
                logger.error("No trees found in the tree model.");
                return false;
            }

            zipOutputStream = new ZipOutputStream(new FileOutputStream(outputZipFile));

            ZipEntry modelEntry = new ZipEntry(MODEL_CONF);
            zipOutputStream.putNextEntry(modelEntry);
            ByteArrayOutputStream byos = new ByteArrayOutputStream();
            JSONUtils.writeValue(new OutputStreamWriter(byos), treeModel);
            zipOutputStream.write(byos.toByteArray());
            IOUtils.closeQuietly(byos);

            ZipEntry treesEntry = new ZipEntry(MODEL_TREES);
            zipOutputStream.putNextEntry(treesEntry);
            DataOutputStream dataOutputStream = new DataOutputStream(zipOutputStream);
            dataOutputStream.writeInt(trees.size());
            for(List<TreeNode> forest : trees) {
                dataOutputStream.writeInt(forest.size());
                for(TreeNode treeNode : forest) {
                    treeNode.write(dataOutputStream);
                }
            }
            IOUtils.closeQuietly(dataOutputStream);
        } catch (IOException e) {
            logger.error("Error occurred when convert the tree model to zip format.", e);
            return false;
        } finally {
            IOUtils.closeQuietly(zipOutputStream);
            IOUtils.closeQuietly(treeModelInputStream);
        }

        return true;
    }

    public boolean convertZipSpecToBinary(File zipSpecFile, File outputGbtFile) {
        ZipInputStream zipInputStream = null;
        FileOutputStream fos = null;

        try {
            zipInputStream = new ZipInputStream(new FileInputStream(zipSpecFile));
            IndependentTreeModel treeModel = null;
            List<List<TreeNode>> trees = null;

            ZipEntry zipEntry = null;
            do {
                zipEntry = zipInputStream.getNextEntry();
                if ( zipEntry != null ) {
                    if ( zipEntry.getName().equals(MODEL_CONF) ) {
                        ByteArrayOutputStream byos = new ByteArrayOutputStream();
                        IOUtils.copy(zipInputStream, byos);
                        treeModel = JSONUtils.readValue(new ByteArrayInputStream(byos.toByteArray()),
                                IndependentTreeModel.class);
                    } else if (zipEntry.getName().equals(MODEL_TREES)) {
                        DataInputStream dataInputStream = new DataInputStream(zipInputStream);
                        int size = dataInputStream.readInt();
                        trees = new ArrayList<List<TreeNode>>(size);
                        for ( int i = 0; i < size; i ++ ) {
                            int forestSize = dataInputStream.readInt();
                            List<TreeNode> forest = new ArrayList<TreeNode>(forestSize);
                            for ( int j = 0; j < forestSize; j ++ ) {
                                TreeNode treeNode = new TreeNode();
                                treeNode.readFields(dataInputStream);
                                forest.add(treeNode);
                            }
                            trees.add(forest);
                        }
                    }
                }
            } while (zipEntry != null);

            if ( treeModel != null && CollectionUtils.isNotEmpty(trees) ) {
                treeModel.setTrees(trees);
                fos = new FileOutputStream(outputGbtFile);
                treeModel.saveToInputStream(fos);
            } else {
                return false;
            }
        } catch (IOException e) {
            logger.error("Error occurred when convert the zip format model to binary.", e);
            return false;
        } finally {
            IOUtils.closeQuietly(zipInputStream);
            IOUtils.closeQuietly(fos);
        }

        return true;
    }
}
