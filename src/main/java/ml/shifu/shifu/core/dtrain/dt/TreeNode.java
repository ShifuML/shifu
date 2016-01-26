/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain.dt;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ml.shifu.guagua.io.Bytable;

/**
 * Wrapper node and tree index. With tree id and node in {@link TreeNode}.
 * 
 * <p>
 * {@link #features} is for sub-sampling of such node. For feature sub-sampling, {@link FeatureSubsetStrategy} includes
 * ALL, HALF and ONETHIRD.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 * 
 * @see Node
 * @see FeatureSubsetStrategy
 */
public class TreeNode implements Bytable {

    /**
     * Tree id
     */
    private int treeId;

    /**
     * Node to be wrappered
     */
    private Node node;

    /**
     * Sub-sampling features.
     */
    private List<Integer> features;

    public TreeNode() {
    }

    public TreeNode(int treeId, Node node) {
        this.treeId = treeId;
        this.node = node;
    }

    public TreeNode(int treeId, Node node, List<Integer> features) {
        this.treeId = treeId;
        this.node = node;
        this.features = features;
    }

    /**
     * @return the treeId
     */
    public int getTreeId() {
        return treeId;
    }

    /**
     * @return the node
     */
    public Node getNode() {
        return node;
    }

    /**
     * @param treeIndex
     *            the treeIndex to set
     */
    public void setTreeId(int treeId) {
        this.treeId = treeId;
    }

    /**
     * @param node
     *            the node to set
     */
    public void setNode(Node node) {
        this.node = node;
    }

    /**
     * @return the features
     */
    public List<Integer> getFeatures() {
        return features;
    }

    /**
     * @param features
     *            the features to set
     */
    public void setFeatures(List<Integer> features) {
        this.features = features;
    }

    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(treeId);
        this.node.write(out);
        if(features == null) {
            out.writeInt(0);
        } else {
            out.writeInt(features.size());
            for(Integer index: features) {
                out.writeInt(index);
            }
        }
    }

    @Override
    public void readFields(DataInput in) throws IOException {
        this.treeId = in.readInt();
        this.node = new Node();
        this.node.readFields(in);

        int len = in.readInt();
        this.features = new ArrayList<Integer>();
        for(int i = 0; i < len; i++) {
            this.features.add(in.readInt());
        }
    }

}
