from array import array
from manim import *
import os
import itertools as it
from manim.mobject import graph

from numpy import number
import numpy as np

class CodeVideo(Scene):
    def construct(self):
        
        self.play(Create(Dot(color=RED_A)), run_time = 4)



class SinAndCosFunctionPlot(Scene):
    def construct(self):
        axes = Axes(
            x_range=[-10, 10.7, 1],
            y_range=[-1.5, 1.5, 1],
            x_length=10,
            axis_config={"color": GREY_A},
            x_axis_config={
                "numbers_to_include": np.arange(-10, 9.01, 2),
                "numbers_with_elongated_ticks": np.arange(-10, 10.01, 2),
            },
            #y_axis_config={
            #    ""
            #}
            tips=True,
        )
        axes_labels = axes.get_axis_labels()
        sin_graph = axes.plot(lambda x: np.sin(x), color=BLUE)
        cos_graph = axes.plot(lambda x: np.cos(x), color=RED)

        sin_label = axes.get_graph_label(
            sin_graph, "\\sin(x)", x_val=-10, direction=UP / 2
        )
        cos_label = axes.get_graph_label(
            cos_graph, label="\\cos(x)"
        )

        vert_line = axes.get_vertical_line(
            axes.i2gp(TAU, cos_graph), color=YELLOW, line_func=Line
        )
        line_label = axes.get_graph_label(
            cos_graph, "x=2\pi", x_val=TAU, direction=UP, color=WHITE
        )

        plot = VGroup(axes, vert_line)
        labels = VGroup(axes_labels, sin_label, cos_label, line_label)
        self.add(plot, labels)
        self.wait(2)
        self.play(Create(sin_graph), Create(cos_graph))
        self.wait()



class Curve_2(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0, 5],
            y_range=[0, 6],
            x_axis_config={
                "numbers_to_include": [1, 2, 3, 4],
                "exclude_origin_tick":True,
            },
            tips=False,
        )

        labels = ax.get_axis_labels()

        curve_1 = ax.plot(lambda x: 4 * x - x ** 2, x_range=[0, 4], color=BLUE_C)
        curve_2 = ax.plot(
            lambda x: 0.8 * x ** 2 - 3 * x + 4,
            x_range=[0, 4],
            color=GREEN,
        )

        line_1 = ax.get_vertical_line(ax.input_to_graph_point(2, curve_1), color=GREY_A)
        line_2 = ax.get_vertical_line(ax.i2gp(3, curve_1), color=GREY_A)
        line_3 = ax.get_vertical_line(ax.i2gp(0.695988, curve_1), color=GREY_A)
        line_4 = ax.get_vertical_line(ax.i2gp(1, curve_1), color=GREY_A)
        

        dx_list = [0.5, 0.25, 0.1, 0.05, 0.025, 0.0001]
        rectangles = VGroup(
            *[
                ax.get_riemann_rectangles(
                    graph = curve_2,
                    x_range=[2, 3],
                    dx=dx,
                    color=[ORANGE, YELLOW],
                    #stroke_color=WHITE,
                    stroke_width=0,
                )
                for dx in dx_list
            ]
        )
        first_area = rectangles[0]
        area_1 = ax.get_area(curve_2, [0.695988, 1], bounded_graph=curve_1, color=PURPLE_B, opacity=0.76)
        self.play(FadeIn(ax), FadeIn(labels))
        self.play(Create(curve_1), Create(curve_2))
        self.play(Write(line_1), Write(line_2), Write(line_3), Write(line_4), run_time = 0.5)
        self.play(FadeIn(first_area), FadeIn(area_1))
        
        for k in range(1, len(dx_list)):
            new_area = rectangles[k]
            self.play(Transform(first_area, new_area), run_time = 1)
            self.wait(0.2)
        
        self.wait()
        



class ManimCELogo(Scene):
    def construct(self):
        self.camera.background_color = "#ece6e2"
        logo_green = "#87c2a5"
        logo_blue = "#525893"
        logo_red = "#e07a5f"
        logo_black = "#343434"
        ds_m = MathTex(r"\mathbb{M}", fill_color=logo_black).scale(7)
        ds_m.shift(2.25 * LEFT + 1.5 * UP)
        circle = Circle(color=logo_green, fill_opacity=1).shift(LEFT)
        square = Square(color=logo_blue, fill_opacity=1).shift(UP)
        triangle = Triangle(color=logo_red, fill_opacity=1).shift(RIGHT)
        logo = VGroup(triangle, square, circle, ds_m)  # order matters
        logo.move_to(ORIGIN)
        
        self.play(GrowFromCenter(logo))
        self.wait()



class BraceAnnotation(Scene):
    def construct(self):
        dot = Dot([-2, -1, 0])
        dot2 = Dot([2, 1, 0])

        line = Line(dot.get_center(), dot2.get_center()).set_color(ORANGE)

        b1 = Brace(line)
        b1text = b1.get_text("Horizontal distance")  #水平距离

        b2 = Brace(line, direction=line.copy().rotate(PI / 2).get_unit_vector())
        b2text = b2.get_tex("x-x_1")
        # self.add(line, dot, dot2, b1, b2, b1text, b2text)
        #g = Group(line, dot, dot2, b1, b2, b1text, b2text)
        #self.play(FadeIn(g))
        self.add(dot, dot2)
        self.play(Create(line))
        self.play(Write(b1), Write(b1text))
        self.wait()
        self.play(Write(b2))
        self.add(b2text)
        self.wait()



class VectorArrow(Scene):
    def construct(self):
        dot = Dot(ORIGIN)
        arrow = Arrow(ORIGIN, [2, 2, 0], buff=0)
        numberplane = NumberPlane()
        origin_text = MathTex(r'\begin{bmatrix}1\\2\end{bmatrix}').next_to(dot, UL)
        tip_text = MathTex('(2, 2)').next_to(arrow.get_end(), RIGHT)
        g = Group(numberplane, dot, arrow, origin_text, tip_text)
        self.play(FadeIn(g))
        self.wait()



class PointMovingOnShapes(Scene):
    def construct(self):
        circle = Circle(radius=1, color=BLUE)
        dot = Dot()
        dot2 = dot.copy().shift(RIGHT)
        line = Line([3, 0, 0], [5, 0, 0])
        
        self.play(GrowFromCenter(circle)) 
        self.add(dot)
        self.add(line)
        self.play(Transform(dot, dot2))
         
        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.play(Rotating(dot, about_point=[2, 0, 0]), run_time=1.5)
        self.wait()



class MovingAngle(Scene):
    def construct(self):
        rotation_center = LEFT

        theta_tracker = ValueTracker(110)
        line1 = Line(LEFT, RIGHT)
        line_moving = Line(LEFT, RIGHT)
        line_ref = line_moving.copy()
        line_moving.rotate(
            theta_tracker.get_value() * DEGREES, about_point=rotation_center
        )
        theta = Angle(line1, line_moving, radius=0.5, other_angle=False)
        tex = MathTex(r"\theta").move_to(
            Angle(
                line1, line_moving, radius=0.5 + 3 * SMALL_BUFF, other_angle=False
            ).point_from_proportion(0.5)
        )

        self.add(line1, line_moving, theta, tex)
        self.wait()

        line_moving.add_updater(
            lambda x: x.become(line_ref).rotate(
                theta_tracker.get_value() * DEGREES, about_point=rotation_center
            )
        )

        theta.add_updater(
            lambda x: x.become(Angle(line1, line_moving, radius=0.5, other_angle=False))
        )
        tex.add_updater(
            lambda x: x.move_to(
                Angle(
                    line1, line_moving, radius=0.5 + 3 * SMALL_BUFF, other_angle=False
                ).point_from_proportion(0.5)
            )
        )

        self.play(theta_tracker.animate.set_value(40))
        self.play(theta_tracker.animate.increment_value(140))
        self.play(tex.animate.set_color(RED), run_time=0.5)
        self.play(theta_tracker.animate.set_value(350))



class MovingGroupToDestination(Scene):
    def construct(self):
        group = VGroup(Dot(LEFT), Dot(ORIGIN), Dot(RIGHT, color=RED), Dot(2 * RIGHT)).scale(1.4)
        dest = Dot([4, 3, 0], color=YELLOW)
        self.add(group, dest)
        self.play(group.animate.shift(dest.get_center() - group[2].get_center()))
        self.wait(0.5)



class MovingFrameBox(Scene):
    def construct(self):
        text=MathTex(
            "\\frac{d}{dx}f(x)g(x)=","f(x)\\frac{d}{dx}g(x)","+",
            "g(x)\\frac{d}{dx}f(x)"
        )
        self.play(Write(text))
        framebox1 = SurroundingRectangle(text[1], buff = .1)
        framebox2 = SurroundingRectangle(text[3], buff = .1)
        self.play(
            Create(framebox1),
        )
        self.wait()
        self.play(
            ReplacementTransform(framebox1,framebox2),
        )
        self.wait()


        
class TexWithSingleStringArrayFail(Scene):
    def construct(self):
        tex_string = "Single string"
        math_text_string = "x + y = 3"

        tex = Tex(tex_string)
        math_text = MathTex(math_text_string)

        logger.info(f"len(tex_string): {len(tex_string)}")             #len(tex_string): 13 
        logger.info(f"len(math_text_string): {len(math_text_string)}") #len(math_text_string): 9
        #print()
        logger.info(f"len(tex): {len(tex)}")                           #len(tex): 1
        logger.info(f"len(math_text): {len(math_text)}")               #len(math_text): 1 

        vg = VGroup(tex,math_text).scale(3).arrange(DOWN)

        self.add(vg)
        self.wait()

class TexWithSingleStringArray(Scene):
    def construct(self):
        tex_string = "Single string"
        math_text_string = "x + y = 3"

        tex = Tex(tex_string)[0] # <- Add [0]
        math_text = MathTex(math_text_string)[0] # <- Add [0]

        logger.info(f"len(tex_string): {len(tex_string)}")#len(tex_string): 13
        logger.info(f"len(math_text_string): {len(math_text_string)}")#len(math_text_string): 9 
        print()
        logger.info(f"len(tex): {len(tex)}")#len(tex): 12
        logger.info(f"len(math_text): {len(math_text)}")#len(math_text): 5 

        vg = VGroup(tex,math_text).scale(3).arrange(DOWN)

        self.add(vg)
        self.wait()
    
def get_tex_indexes(
    tex,
    number_config={"height":0.28},
    colors=[RED,TEAL,PURPLE,GREEN,BLUE],
    funcs=[lambda mob,tex: mob.next_to(tex,DOWN,buff=0.2)]
    ):
    numbers = VGroup()
    colors = it.cycle(colors)
    for i,s in enumerate(tex):
        n = Text(f"{i}",color=next(colors),**number_config)
        for f in funcs:
            f(n,s)
        numbers.add(n)
    return numbers



class ShowIndexesOfTex(Scene):
    def construct(self):
        tex_string = "Single string"
        math_text_string = "x + y = 3"

        tex = Tex(tex_string) # <- Add [0]
        math_text = MathTex(math_text_string) # <- Add [0]

        vg = VGroup(tex,math_text).scale(3).arrange(DOWN,buff=1)

        n1 = get_tex_indexes(tex[0])
        n2 = get_tex_indexes(math_text[0])

        tex[0][3].set_color(ORANGE)
        math_text[0][4].set_color(ORANGE)

        self.add(vg,n1,n2)
        self.wait()

class MultipleTexString(Scene):
    def construct(self):
        tex_string = ["Multiple ","tex ","string"]
        math_text_string = ["x+","y","=","3"]

        tex = Tex(*tex_string) # <- Add [0]
        math_text = MathTex(*math_text_string) # <- Add [0]

        vg = VGroup(tex,math_text).scale(3).arrange(DOWN,buff=1)

        n1 = get_tex_indexes(tex)
        n2 = get_tex_indexes(math_text)

        f = lambda mob,tex: mob.next_to(tex,UP,buff=0)
        n_1_1 = get_tex_indexes(tex[0],funcs=[f])
        n_1_2 = get_tex_indexes(tex[1],funcs=[f])
        n_1_3 = get_tex_indexes(tex[2],funcs=[f])

        tex[0][2].set_color(TEAL)
        tex[1][1].set_color(ORANGE)
        tex[2][3].set_color(PINK)

        math_text[0].set_color(PURPLE)

        self.add(vg,n1,n2,n_1_1,n_1_2,n_1_3)
        self.wait()



class CodeFromString(Scene):
    def construct(self):
        code = '''
        from manim import Scene, Square

        class FadeInSquare(Scene):
            def construct(self):
                s = Square()
                self.play(FadeIn(s))
                self.play(s.animate.scale(3))
                self.wait()'''
        rendered_code = Code(
            code=code,
            tab_width=4,
            background="window",
            language="Python",
            font="Monospace",
            style="monokai"
        )
        self.draw_code_all_lines_at_a_time(rendered_code)
        self.wait()

    def draw_code_all_lines_at_a_time(self, code, **kwargs):
        self.play(LaggedStart(*[
                Write(code[i])
                for i in range(len(code))
            ]),
            **kwargs
        )


class RotationUpdater(Scene):
    def construct(self):
        def updater_forth(mobj, dt):
            mobj.rotate_about_origin(- dt * PI / 30) #调整速度使其像秒针一样随时间运动
        def updater_back(mobj, dt):
            mobj.rotate_about_origin(dt)
        dot = Dot(ORIGIN, color=WHITE).scale(0.7)
        line_reference = Line(ORIGIN, LEFT).set_color(WHITE)
        line_moving = Line(ORIGIN, LEFT).set_color(YELLOW)
        line_moving.add_updater(updater_forth)
        self.add(line_reference, line_moving)
        self.add(dot)
        self.wait(60) #模拟秒针运动一分钟的动画
        line_moving.remove_updater(updater_forth)
        line_moving.add_updater(updater_back)
        self.wait(6)
        line_moving.remove_updater(updater_back)
        self.wait(0.5)


class GraphingMovement(Scene):
    def construct(self):
        
        axes = Axes(
            x_range=[0, 5],
            y_range=[0, 3],
            x_length=5,
            y_length=3,
            axis_config={"numbers_to_exclude": []},
            #tips=True,
        ).add_coordinates()
        axes.to_edge(UR)
        axes_labels = axes.get_axis_labels(x_label="x", y_label="f(x)")

        graph = axes.plot(lambda x: x**0.5, x_range=[0, 4], color = YELLOW)
        graphing_stuff = VGroup(axes, graph, axes_labels)

        self.play(DrawBorderThenFill(axes), Write(axes_labels))
        self.play(Create(graph))
        self.play(graphing_stuff.animate.shift(DOWN * 4))
        self.play(graphing_stuff.animate.shift(LEFT * 3))



class Tute(Scene):
    def construct(self):
        
        e = ValueTracker(0.01)
        plane = PolarPlane(
            radius_max = 3, 
            azimuth_units = "PI radians", 
            azimuth_step = 12,
            azimuth_label_font_size = 36
        ).add_coordinates()
        plane.shift(LEFT * 3)
        plane.scale(0.8)
        graph1 = always_redraw(
            lambda : ParametricFunction(
                lambda t: plane.polar_to_point(2 * np.sin(3*t), t),
                t_range=[0, e.get_value()], 
                color=GREEN
            )
        )
        dot1 = always_redraw(
            lambda :Dot(color = GREEN, fill_opacity = 0.8).scale(0.5).move_to(graph1.get_end())
        )

        axes = Axes(
            x_range=[0, 4],
            x_length=6,
            y_range=[-3, 3],
            y_length=6,
            tips=False
        ).shift(RIGHT * 4)
        axes.scale(0.7)
        axes.add_coordinates()
        graph2 = always_redraw(
            lambda : axes.plot(
                lambda x: 2*np.sin(3*x),
                x_range=[0, e.get_value()],
                color = GREEN
            )
        )
        dot2 = always_redraw(
            lambda : Dot(color=GREEN, fill_opacity=0.8).scale(0.5).move_to(graph2.get_end())
        )

        title = MathTex("f(\\theta) = 2sin(3\\theta)", color = GREEN).next_to(plane, UP, buff=0.3)
        title.scale(0.8)

        self.play(LaggedStart(
            Write(plane), Create(axes), Write(title),
            run_time = 3, lag_ratio = 0.5
        ))
        self.add(graph1, graph2, dot1, dot2)
        self.play(e.animate.set_value(PI), run_time = 6, rate_functions = linear)
        self.wait()



class PointWithTrace(Scene):
    def construct(self):
        
        path = VMobject()
        dot = Dot()
        path.set_points_as_corners([dot.get_center(), dot.get_center()])

        def update_path(path):
            previous_path = path.copy()
            previous_path.add_points_as_corners([dot.get_center()])
            path.become(previous_path)
        path.add_updater(update_path)
        self.add(path, dot)
        self.play(Rotating(dot, radians = PI, about_point=RIGHT, runtime = 2))
        self.play(dot.animate.shift(UP))
        self.play(dot.animate.shift(LEFT))
        self.wait()



class SineCurveUnitCircle(Scene):
    def construct(self):
        self.show_axis()
        self.show_circle()
        self.move_dot_and_draw_curve()
        self.wait()
    
    def show_axis(self):
        x_start = np.array([-6, 0, 0])
        x_end = np.array([6, 0, 0])

        y_start = np.array([-4, -2, 0])
        y_end = np.array([-4, 2, 0])

        x_axis = Line(x_start, x_end)
        y_axis = Line(y_start, y_end)

        self.add(x_axis, y_axis)
        self.add_x_labels()

        self.origin_point = np.array([-4, 0, 0])#设置两个后面会用到的点，给到self方便后面调用（公用），不用重新输入
        self.curve_start = np.array([-4, 0, 0])
    
    def add_x_labels(self):
        x_labels = [
            MathTex("\\pi"), MathTex("2\\pi"),
            MathTex("3\\pi"), MathTex("4\\pi"),   
        ]
        for i in range(len(x_labels)):
            x_labels[i].next_to(np.array([-2 + 2*i, 0, 0]), DOWN)#依次写入坐标
            self.add(x_labels[i])
    
    def show_circle(self):
        circle = Circle(radius=1).move_to(self.origin_point)
        self.add(circle)
        self.circle = circle
    
    def move_dot_and_draw_curve(self):
        orbit = self.circle   #orbit：轨道
        origin_point = self.origin_point

        dot = Dot(radius=0.08, color=YELLOW)
        dot.move_to(orbit.point_from_proportion(0))#获取沿VMobject路径按一定比例的点。

        self.t_offset = 0
        rate = 0.25   #比例系数，使点在1s内转过1/4圆

        def go_around_circle(mob, dt):
            self.t_offset += (dt * rate)
            mob.move_to(orbit.point_from_proportion(self.t_offset % 1))#这个地方有点妙，它使得可以以1/rate为周期

        def get_line_to_circle():
            return Line(origin_point, dot.get_center(), color = BLUE)

        def get_line_to_curve():#始终找两点新位置连成线段
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            return Line(dot.get_center(), np.array([x, y, 0]), color = YELLOW_A, stroke_width = 2)

        self.curve = VGroup()
        self.curve.add(Line(self.curve_start, self.curve_start))


        def get_curve():#以小线段组成需要的函数曲线
            last_line = self.curve[-1]#-1也许是指倒数第一个
            x = self.curve_start[0] + self.t_offset * 4
            y = dot.get_center()[1]
            new_line = Line(last_line.get_end(), np.array([x, y, 0]), color = YELLOW_D)
            self.curve.add(new_line)

            return self.curve

        dot.add_updater(go_around_circle)

        origin_to_circle_line = always_redraw(get_line_to_circle)
        dot_to_curve_line = always_redraw(get_line_to_curve)
        sine_curve_line = always_redraw(get_curve)

        self.add(dot)
        self.add(orbit, origin_to_circle_line, dot_to_curve_line, sine_curve_line)
        self.wait(8)

        dot.remove_updater(go_around_circle)


class MovingZoomedSceneAround(ZoomedScene):
    def __init__(self, **kwargs):
        ZoomedScene.__init__(
            self, 
            zoom_factor = 0.3,
            zoomed_display_height = 3,
            zoomed_display_width = 4,
            image_frame_stroke_width = 20,
            zoomed_camera_config={
                "default_frame_stroke_width":3
            },
            **kwargs
        )
    
    def construct(self):
        dot = Dot().shift(UL * 2)
        image = ImageMobject(np.uint8([[0, 100, 30, 200],
                                       [255, 0, 5, 33]]))
        image.height = 7

        self.add(image, dot)
        
        zoomed_camera = self.zoomed_camera
        zoomed_display = self.zoomed_display
        frame = zoomed_camera.frame
        zoomed_display_frame = zoomed_display.display_frame

        frame.move_to(dot)
        frame.set_color(PURPLE)
        zoomed_display_frame.set_color(RED)
        zoomed_display.shift(DOWN)

        zd_rect = BackgroundRectangle(zoomed_display, fill_opacity=0, buff=MED_SMALL_BUFF)
        self.add_foreground_mobject(zd_rect)

        unfold_camera = UpdateFromFunc(zd_rect, lambda rect: rect.replace(zoomed_display))


        self.play(Create(frame))
        self.activate_zooming()#此方法用于激活照相机的缩放。

        self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera)#这是显示缩放摄影机内容的迷你显示屏弹出的动画。
        # Scale in        x   y   z
        scale_factor = [0.5, 1.5, 0]#缩放摄像机大小
        self.play(
            frame.animate.scale(scale_factor),
            zoomed_display.animate.scale(scale_factor),
        )
        #self.wait()
        #self.play(ScaleInPlace(zoomed_display, 2))
        #self.wait()
        self.play(frame.animate.shift(2.5 * DOWN))
        #self.wait()
        self.play(self.get_zoomed_display_pop_out_animation(), unfold_camera, rate_func=lambda t: smooth(1 - t))
        self.play(Uncreate(zoomed_display_frame), FadeOut(frame))
        self.wait()



class TexAndMathTextColors(Scene):
    def construct(self):
        # If the texts are simple, that is, 
        # without fractions, roots, subscripts and/or superscripts,
        # it is possible to color the text as follows.
        tex = Tex(
            r"This is \LaTeX, with a formula: $x^2$",
            tex_to_color_map={
                r"\LaTeX": RED,
                "formula": ORANGE,
                "$x^2$": TEAL,
            }
        )
        math_tex = MathTex(
            r"\frac{\rm d}{\rm d\it x}f = f'(x)",
            # You cannot use "d": RED, try it
            tex_to_color_map={
                "f'(x)": YELLOW,
                # "d": RED,
            }
        )
        text = Text(
            "Normal text with PC fonts",
            font="Arial",
            # These arguments can present problems in
            # versions prior to 0.5.0, MarkupText class
            # can be used instead of Text if text with
            # alot of decorations is needed.
            # Use Text for simple texts.
            t2c={
                "Normal": RED,
            },
            t2w={
                "text": BOLD,
            },
            t2s={
                "fonts": ITALIC
            }
        )

        Group(tex, math_tex, text).set(width=config.frame_width-1).arrange(DOWN)

        self.add(tex, math_tex, text)
        self.wait()


class MultipleTexString(Scene):
    def construct(self):
        tex_string = ["Multiple ","tex ","string"]
        math_text_string = ["x+","y","=","3"]
        

        tex = Tex(*tex_string) # <- Add [0]
        math_text = MathTex(*math_text_string) # <- Add [0]

        vg = VGroup(tex,math_text).scale(3).arrange(DOWN,buff=1)

        n1 = get_tex_indexes(tex)
        n2 = get_tex_indexes(math_text)

        f = lambda mob,tex: mob.next_to(tex,UP,buff=0)
        n_1_1 = get_tex_indexes(tex[0],funcs=[f])
        n_1_2 = get_tex_indexes(tex[1],funcs=[f])
        n_1_3 = get_tex_indexes(tex[2],funcs=[f])

        tex[0][2].set_color(TEAL)
        tex[1][1].set_color(ORANGE)
        tex[2][3].set_color(PINK)

        math_text[0].set_color(PURPLE)

        self.add(vg,n1,n2,n_1_1,n_1_2,n_1_3)
        self.wait()