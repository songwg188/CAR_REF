def set_template(args):
    args.model = args.template
    if args.qm and args.qv:
        raise ValueError('qm and qv should not be used at the same time!')

    if args.qm or args.qv:
        if args.n_colors == 1:
            args.in_channel = 2
            args.in_glo_channel = 1
        if args.n_colors == 3:
            if args.qm:
                args.in_channel = 5
            elif args.qv:
                args.in_channel = 4
            args.in_glo_channel =3
    else:
        args.in_channel = args.n_colors
    
    if args.template.find('qmar') >= 0:
        args.model = 'qmar'
        args.n_feats = 64
        args.res_scale = 0.1

    if args.template.find('qmarg') >= 0:
        args.model = 'qmarg'
        args.n_feats = 64
        args.res_scale = 0.1

    if args.template.find('drunet') >=0:
        args.model = 'drunet'
        if args.n_colors == 1:
            args.in_channel = 2
        if args.n_colors == 3:
            args.in_channel = 4