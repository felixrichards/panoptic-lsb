class_maps = {
    'basic': {
        'idxs': [
            2, 2, 2, 2,
            1, 1, 1, 1, 3,
            0, 0, 0, 0, 0,
            0, 4, 4, 4, 0
        ],
        'classes': [
            'None', 'Galaxy', 'Fine structures', 'Diffuse halo', 'Contaminants'
        ],
        'class_balances': [
            1., 1., 1.
        ],
        'split_components': [
            {'split': True, 'blur': 0, 'prune': True},
            {'split': True, 'blur': 199, 'prune': True},
            {'split': True, 'blur': 0, 'prune': True},
            {'split': False, 'prune': True}
        ],
        'aggregate_methods': [
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
        ],
        'segment': [
            True, True, True
        ],
        'detect': [
            True, True, False
        ],
    },
    'basicnocontaminants': {
        'idxs': [
            0, 2, 2, 2,
            1, 1, 1, 1, 3,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        ],
        'classes': [
            'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo'
        ],
        'class_balances': [
            1., 1., 1.
        ],
        'split_components': [
            {'split': True, 'blur': 0, 'prune': True},
            {'split': True, 'blur': 199, 'prune': True},
            {'split': True, 'blur': 0, 'prune': True},
        ],
        'aggregate_methods': [
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
            {'method': 'nms', 'threshold': .3},
        ],
        'segment': [
            True, True, True
        ],
        'detect': [
            True, True, True
        ],
    },
    'basicnocontaminantsnocompanions': {
        'idxs': [
            0, 2, 2, 2,
            1, 0, 3, 0, 3,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        ],
        'classes': [
            'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo'
        ],
        'class_balances': [
            1., 1., 1.
        ],
        'split_components': [
            {'split': True, 'blur': 0, 'prune': True},
            {'split': True, 'blur': 199, 'prune': True},
            {'split': True, 'blur': 0, 'prune': True},
        ],
        'aggregate_methods': [
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
            {'method': 'nms', 'threshold': .3},
        ],
        'segment': [
            True, True, True
        ],
        'detect': [
            True, True, True
        ],
    },
    'basiccirrusnocompanions': {
        'idxs': [
            0, 2, 2, 2,
            1, 0, 3, 0, 3,
            0, 0, 0, 0, 0,
            0, 0, 0, 4, 0
        ],
        'classes': [
            'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo', 'Cirrus'
        ],
        'class_balances': [
            1., 1., 1., 1.
        ],
        'split_components': [
            {'split': True, 'blur': 0, 'prune': True},
            {'split': True, 'blur': 199, 'prune': True},
            {'split': True, 'blur': 0, 'prune': True},
            {'split': False, 'prune': True},
        ],
        'aggregate_methods': [
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
        ],
        'segment': [
            True, True, True, True
        ],
        'detect': [
            True, True, True, False
        ],
    },
    'basichalosnocompanions': {
        'idxs': [
            0, 2, 2, 2,
            1, 0, 3, 0, 3,
            0, 0, 0, 0, 0,
            0, 0, 4, 0, 0
        ],
        'classes': [
            'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo', 'Ghosted halo'
        ],
        'class_balances': [
            1., 1., 1., 1.
        ],
        'split_components': [
            {'split': True, 'blur': 0, 'prune': True},
            {'split': True, 'blur': 199, 'prune': True},
            {'split': True, 'blur': 0, 'prune': True},
            {'split': True, 'blur': 0, 'prune': True}
        ],
        'aggregate_methods': [
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
        ],
        'segment': [
            True, True, True, True
        ],
        'detect': [
            True, True, True, True
        ],
    },
    'basichaloscirrusnocompanions': {
        'idxs': [
            0, 2, 2, 2,
            1, 0, 3, 0, 3,
            0, 0, 0, 0, 0,
            0, 0, 4, 5, 0
        ],
        'classes': [
            'None', 'Galaxy', 'Elongated tidal structures', 'Diffuse halo', 'Ghosted halo', 'Cirrus'
        ],
        'class_balances': [
            1., 1., 1., 1., 1.
        ],
        'split_components': [
            {'split': True, 'blur': 0, 'prune': True},
            {'split': True, 'blur': 199, 'prune': True},
            {'split': True, 'blur': 0, 'prune': True},
            {'split': True, 'blur': 0, 'prune': True},
            {'split': False, 'prune': True},
        ],
        'aggregate_methods': [
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
            {'method': 'nms', 'threshold': .3},
            {'method': 'union'},
            {'method': 'union'},
        ],
        'segment': [
            True, True, True, True, True
        ],
        'detect': [
            True, True, True, True, False
        ],
    },
    'streamstails': {
        'idxs': [
            0, 0, 1, 2,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0
        ],
        'classes': [
            'None', 'Tidal tails', 'Streams'
        ],
        'class_balances': [
            1., 1.
        ],
        'split_components': [
            {'split': True, 'blur': 199, 'prune': True},
            {'split': True, 'blur': 199, 'prune': True},
        ],
        'aggregate_methods': [
            {'method': 'union'},
            {'method': 'union'},
        ],
        'segment': [
            True, True
        ],
        'detect': [
            True, True
        ],
    },
    'all': {
        'idxs': [
            5, 0, 3, 4,
            1, 0, 6, 0, 2,
            0, 0, 0, 0, 0,
            0, 9, 7, 8, 0
        ],
        'classes': [
            'None', 'Main galaxy', 'Halo', 'Tidal tails', 'Streams', 'Shells', 'Companion', 'Ghosted halo', 'Cirrus',
            'High background'
        ],
        'class_balances': [
            1., 1., 1., 1., 1., 1., 1., 1., 1.
        ],
        'split_components': [
            True, True, True, True, False, True, True, False, False
        ]
    },
    'cirrus': {
        'idxs': [
            1
            # 0, 0, 0, 0,
            # 0, 0, 0, 0, 0,
            # 0, 0, 0, 0, 0,
            # 0, 0, 0, 1, 0
        ],
        'classes': [
            'None', 'Cirrus'
        ],
        'class_balances': [
            1.
        ],
        'split_components': [
            False
        ],
        'segment': [
            True
        ],
        'detect': [
            False
        ],
    },
}
