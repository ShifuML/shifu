/*
 * Copyright 2016 PayPal Software Foundation
 */
package ml.shifu.shifu.core.dtrain.dt;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import ml.shifu.guagua.io.Bytable;

/**
 * {@link SimpleBitSet} mocks part of functions in {@link BitSet}. The reason for {@link SimpleBitSet} is to implement
 * compressed serialization format in {@link Bytable}, which can help compress binary tree ensemble model.
 * 
 * @author pengzhang
 */
class SimpleBitSet<T> implements Set<T>, Bytable {

    private byte[] words;

    public SimpleBitSet() {
        // default is one long
        this.words = new byte[8];
    }

    public SimpleBitSet(int bitLen) {
        words = new byte[bitLen / 8 + 1];
    }

    public SimpleBitSet(int bitLen, SimpleBitSet<? extends T> sbs) {
        words = Arrays.copyOf(sbs.words, sbs.words.length);;
    }

    @Override
    public int size() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean isEmpty() {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean contains(Object o) {
        if(!(o != null && o instanceof Number)) {
            throw new IllegalArgumentException("Input is not a number");
        }
        int intValue = ((Number) o).intValue();
        int wordIndex = wordIndex(intValue);
        if(wordIndex >= this.words.length) {
            return false;
        }
        int bitIndex = intValue % 8;
        return ((words[wordIndex] & (1 << bitIndex)) != 0);
    }

    /**
     * Given a bit index, return word index containing it.
     */
    private static int wordIndex(int bitIndex) {
        return bitIndex >> 3;
    }

    @Override
    public Object[] toArray() {
        throw new UnsupportedOperationException();
    }

    @SuppressWarnings("hiding")
    @Override
    public <T> T[] toArray(T[] a) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean add(T e) {
        if(e == null || !(e instanceof Number)) {
            throw new IllegalArgumentException("Input is not a number");
        }
        int intValue = ((Number) e).intValue();
        // if(intValue >= this.words.length * 8) {
        // words = Arrays.copyOf(words, Math.max(2 * words.length, intValue / 8));
        // }
        int byteIndex = wordIndex(intValue);
        while(byteIndex >= this.words.length) {
            words = Arrays.copyOf(words, 2 * words.length);
        }
        int bitIndex = intValue % 8;
        words[byteIndex] |= (1 << bitIndex);
        return true;
    }

    @Override
    public boolean remove(Object o) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean containsAll(Collection<?> c) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean retainAll(Collection<?> c) {
        throw new UnsupportedOperationException();
    }

    @Override
    public boolean removeAll(Collection<?> c) {
        throw new UnsupportedOperationException();
    }

    @Override
    public void clear() {
        throw new UnsupportedOperationException();
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#write(java.io.DataOutput)
     */
    @Override
    public void write(DataOutput out) throws IOException {
        out.writeInt(this.words.length);
        for(byte by: this.words) {
            out.writeByte(by);
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.guagua.io.Bytable#readFields(java.io.DataInput)
     */
    @Override
    public void readFields(DataInput in) throws IOException {
        int len = in.readInt();
        this.words = new byte[len];
        for(int i = 0; i < len; i++) {
            this.words[i] = in.readByte();
        }
    }

    @Override
    public String toString() {
        List<Short> shortList = new ArrayList<Short>();
        for(int i = 0; i < words.length * 8; i++) {
            if(contains((short) i)) {
                shortList.add((short) i);
            }
        }
        return "SimpleBitSet [" + shortList.toString() + "]";
    }

    @SuppressWarnings({ "unchecked", "rawtypes" })
    @Override
    public Iterator<T> iterator() {
        List list = new ArrayList();
        for(short i = 0; i < words.length * 8; i++) {
            if(contains(i)) {
                list.add(i);
            }
        }
        return list.iterator();
    }

    @Override
    public boolean addAll(Collection<? extends T> c) {
        throw new UnsupportedOperationException();
    }

    public static void main(String[] args) {
        SimpleBitSet<Short> sbs = new SimpleBitSet<Short>(2);

        sbs.add((short) 1);
        sbs.add((short) 2);
        sbs.add((short) 3);
        sbs.add((short) 4);
        sbs.add((short) 5);
        sbs.add((short) 6);
        sbs.add((short) 7);
        sbs.add((short) 8);
        sbs.add((short) 9);
        sbs.add((short) 10);
        sbs.add((short) 1);
        sbs.add((short) 2);
        sbs.add((short) 1);
        sbs.add((short) 255);

        System.out.println(sbs.contains(1));
        System.out.println(sbs.contains(2));
        System.out.println(sbs.contains(8));
        System.out.println(sbs.contains(0));
        System.out.println(sbs.contains(12));
        System.out.println(sbs.contains((short) 1));
        System.out.println(sbs.contains((short) 255));
        System.out.println(sbs.contains((short) 258));

        System.out.println(sbs.toString());
        System.out.println("----------------------");

        Iterator<Short> it = sbs.iterator();
        while(it.hasNext()) {
            System.out.println(it.next());
        }

    }

}
