from wtforms import Form, FloatField, SelectField, validators


class InputForm(Form):
    a = SelectField(u'Clump Thickness',
                    choices=[(1, '1'),
                             (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'),
                             (10, '10'), ],
                    coerce=int,
                    validators=[validators.optional()])
    b = SelectField(u'Uniformity of Cell Size',
                    choices=[(1, '1'),
                             (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'),
                             (10, '10'), ],
                    coerce=int,
                    validators=[validators.optional()])
    c = SelectField(u'Uniformity of Cell Shape',
                    choices=[(1, '1'),
                             (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'),
                             (10, '10'), ],
                    coerce=int,
                    validators=[validators.optional()])
    d = SelectField(u'Marginal Adhesion',
                    choices=[(1, '1'),
                             (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'),
                             (10, '10'), ],
                    coerce=int,
                    validators=[validators.optional()])
    e = SelectField(u'Single Epithelial Cell Size          ',
                    choices=[(1, '1'),
                             (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'),
                             (10, '10'), ],
                    coerce=int,
                    validators=[validators.optional()])
    z = SelectField(u'Bare Nuclei',
                    choices=[(1, '1'),
                             (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'),
                             (10, '10'), ],
                    coerce=int,
                    validators=[validators.optional()])
    g = SelectField(u'Bland Chromatin',
                    choices=[(1, '1'),
                             (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'),
                             (10, '10'), ],
                    coerce=int,
                    validators=[validators.optional()])
    h = SelectField(u'Normal Nucleoli',
                    choices=[(1, '1'),
                             (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'),
                             (10, '10'), ],
                    coerce=int,
                    validators=[validators.optional()])
    i = SelectField(u'Mitoses',
                    choices=[(1, '1'),
                             (2, '2'), (3, '3'), (4, '4'), (5, '5'), (6, '6'), (7, '7'), (8, '8'), (9, '9'),
                             (10, '10'), ],
                    coerce=int,
                    validators=[validators.optional()])
