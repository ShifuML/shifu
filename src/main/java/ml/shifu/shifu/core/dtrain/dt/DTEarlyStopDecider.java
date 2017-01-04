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

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link DTEarlyStopDecider} monitor the train error and validation error in the training process. When it
 * identified if the training is over fit or the effort is not worth more training loop, method {@link #add} will
 * return true.
 * <p>
 * {@link MinAverageDecider} 1, Filter Algorithm After I tried several well-known filter algorithm, it all not as good
 * as I want. I tried plot these data by python, and found these training error data varies in loop by tree depth. On
 * the other hand, we have a tendency to get the minimal training error, So I divided these error data into different
 * window by the size of tree depth. And for each window picks the minimum value as the value represent value.
 * <p>
 * Basic on the upper sample values, I adopt a further more filter algorithm: Recursive filtering. Use the average of
 * the last queue size value as the value for the new value and insert into the queue for further operation.
 * <p>
 * 2, Iteration Gain Iteration gain is the reduce value of the error for each loop.
 * <p>
 * {@link #canStop()} 1, Identify the training is over fit When the validation iteration gain is negative for continue 3
 * times, we consider this algorithm is over fitted.
 * <p>
 * 2, Identify not worth more training iteration If the iteration gain is less than one tenth of the max gain value for
 * continue 3 times, we consider it worth no more iteration.
 * 
 * @author haifwu
 */
class DTEarlyStopDecider {

    private static final Logger LOG = LoggerFactory.getLogger(DTEarlyStopDecider.class);
    
    /**
     * if 3 times continue reach the stop requirements, decider will make stop decision
     */
    private static final int MAGIC_NUMBER = 3;

    /**
     * Threshhold value to stop iteration
     */
    private static final double NEARLY_ZERO = 0.000001;
    
    /**
     * Max Depth of a tree
     */
    private int treeDepth;

    /**
     * {@link #validationErrorDecider}, return a positive or negative sign whether training is over fitted.
     */
    private MinAverageDecider validationErrorDecider;
    
    /**
     * Continue count of positive sign over fit
     */
    private int validationGainContinueNearZeroCount;

    public DTEarlyStopDecider(int treeDepth) {
        if(treeDepth <= 0) {
            throw new IllegalArgumentException("Tree num should not be less or equal than zero!");
        }

        this.treeDepth = treeDepth;

        validationErrorDecider = new MinAverageDecider(this.treeDepth, this.treeDepth * MAGIC_NUMBER) {
            @Override
            public boolean getDecide() {
                return this.gain < NEARLY_ZERO;
            }
        };
    }

    /**
     * Add new iteration's train error and validation error into the decider.
     * 
     * @param validationError
     *            validation error
     * @return true if no more iteration needed, else false
     */
    public boolean add(double validationError) {

        boolean validationDecideReady = this.validationErrorDecider.add(validationError);

        if(validationDecideReady) {
            if(validationErrorDecider.getDecide()) {
                this.validationGainContinueNearZeroCount += 1;
                LOG.warn("Continue {} positive sign for not worth more iteration!", this.validationGainContinueNearZeroCount);
            } else {
                this.validationGainContinueNearZeroCount = 0;
            }
        }

        return canStop();
    }

    private boolean canStop() {
        return this.validationGainContinueNearZeroCount >= MAGIC_NUMBER;
    }

    static abstract class MinAverageDecider {

        /**
         * minQueue to get the minimal value of a queue size values
         */
        private final ThreadLocal<MinQueue> minQueue;

        /**
         * Max gain so far
         */
        double maxGain;

        /**
         * Current gain
         */
        double gain;

        /**
         * averageQueue, insert with recursive average value into the queue, and get iteration gain
         */
        private AverageQueue averageQueue;

        MinAverageDecider(int minQueueNum, int averageQueueNum) {
            minQueue = new ThreadLocal<MinQueue>();
            this.minQueue.set(new MinQueue(minQueueNum));
            this.averageQueue = new AverageQueue(averageQueueNum);
            this.maxGain = 0;
        }

        /**
         * Add a value into the decider
         * 
         * @param element
         *            the value to insert to the decide
         * @return true if new gain generated, and decide is ready to get
         */
        public boolean add(double element) {
            if(!this.minQueue.get().add(element)) {
                return false;
            }
            LOG.debug("MinQueue is full, get min value: {}", element);
            if(!this.averageQueue.add(element)) {
                return false;
            }
            this.gain = this.averageQueue.getGain();
            LOG.debug("Average Queue is full, get gain value: {}", this.gain);
            if(this.gain > this.maxGain) {
                this.maxGain = this.gain;
            }
            return true;
        }

        public abstract boolean getDecide();
    }

    /**
     * Generate minimal value of each {@link #capacity} values.
     */
    private static class MinQueue {

        /**
         * total element in current queue
         */
        private int size;

        /**
         * min value in current queue
         */
        private double min;

        /**
         * max capacity of the queue
         */
        private int capacity;

        MinQueue(int capacity) {
            this.capacity = capacity;
            this.clear();
        }

        private void clear() {
            this.min = Double.MAX_VALUE;
            this.size = 0;
        }

        /**
         * Add an element to the queue
         * 
         * @param element
         *            the value to add
         * @return true if the queue is full, and ready to generate the minimal value of this window
         */
        public boolean add(double element) {
            if(element < this.min) {
                this.min = element;
            }
            this.size += 1;

            return this.size >= this.capacity;
        }

        /**
         * Get the minimal value in the queue. Should be called when the queue is full.
         * 
         * @return the value of the minimal value of the queue
         */
        @SuppressWarnings("unused")
        double getQueueMin() {
            double queueMin = this.min;
            this.clear();
            return queueMin;
        }
    }

    /**
     * Generate recursive average value gain.
     */
    private static class AverageQueue {

        /**
         * The max capacity of this queue
         */
        private int capacity;

        /**
         * Array to store values in the queue
         */
        private double[] queueArray;

        /**
         * Total count of value have into queue
         */
        private long totalCount;

        /**
         * Total sum of value in queue
         */
        private double sum;

        AverageQueue(int capacity) {
            this.capacity = capacity;
            this.queueArray = new double[this.capacity];
            this.totalCount = 0;
            this.sum = 0;
        }

        /**
         * Add element into the queue
         * 
         * @param element
         *            the element inset into the queue
         * @return false if the queue is not reach {@link #capacity} yet, else true
         */
        public boolean add(double element) {
            int index = (int) this.totalCount % this.capacity;
            this.totalCount += 1;
            if(this.totalCount <= this.capacity) {
                // Before queue full, we calculate the sum value by add new value
                this.sum += element;
                this.queueArray[index] = this.sum / this.totalCount;
                return false;
            } else {
                // After queue full, we calculate the sum value by add new value minus old value
                this.sum += element - this.queueArray[index];
                this.queueArray[index] = this.sum / this.capacity;
                return true;
            }
        }

        /**
         * Get the iteration gain of current insert value
         * 
         * @return the iteration gain of the current value
         */
        double getGain() {
            int curIndex = (int) (this.totalCount - 1) % this.capacity;
            int lastIndex = (int) (this.totalCount - 2) % this.capacity;
            return this.queueArray[lastIndex] - this.queueArray[curIndex];
        }
    }
}
