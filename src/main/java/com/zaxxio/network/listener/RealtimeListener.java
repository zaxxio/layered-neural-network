package com.zaxxio.network.listener;

import com.zaxxio.network.MultiLayerNetwork;

@FunctionalInterface
public interface RealtimeListener {

    void onUpdate(
            MultiLayerNetwork network,
            RealtimeEvent event
    );
}