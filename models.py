  
  
 f r o m   t e n s o r f l o w . k e r a s . m o d e l s   i m p o r t   S e q u e n t i a l  
 f r o m   t e n s o r f l o w . k e r a s . l a y e r s   i m p o r t   C o n v 2 D  
 f r o m   t e n s o r f l o w . k e r a s . l a y e r s   i m p o r t   M a x P o o l i n g 2 D  
 f r o m   t e n s o r f l o w . k e r a s . l a y e r s   i m p o r t   A v e r a g e P o o l i n g 2 D  
 f r o m   t e n s o r f l o w . k e r a s . l a y e r s   i m p o r t   D e n s e  
 f r o m   t e n s o r f l o w . k e r a s . l a y e r s   i m p o r t   D r o p o u t  
 f r o m   t e n s o r f l o w . k e r a s . l a y e r s   i m p o r t   F l a t t e n  
 i m p o r t   t e n s o r f l o w   a s   t f  
  
  
  
 d e f   V G G _ 1 ( ) :  
         m o d e l 3   =   S e q u e n t i a l ( )  
         m o d e l 3 . a d d ( C o n v 2 D ( 3 2 ,   ( 5 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   d a t a _ f o r m a t = ' c h a n n e l s _ l a s t ' ,   p a d d i n g = ' s a m e ' ,   i n p u t _ s h a p e = ( 6 4 ,   3 2 ,   3 ) ) )  
         m o d e l 3 . a d d ( C o n v 2 D ( 3 2 ,   ( 5 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 2 ,   p a d d i n g = ' s a m e ' ) )  
         m o d e l 3 . a d d ( t f . k e r a s . l a y e r s . B a t c h N o r m a l i z a t i o n ( ) )  
         m o d e l 3 . a d d ( A v e r a g e P o o l i n g 2 D ( ( 3 ,   3 ) ) )  
         m o d e l 3 . a d d ( D r o p o u t ( 0 . 2 ) )  
         m o d e l 3 . a d d ( C o n v 2 D ( 6 4 ,   ( 3 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 1 ,   p a d d i n g = ' s a m e ' ) )  
         m o d e l 3 . a d d ( C o n v 2 D ( 6 4 ,   ( 3 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 1 ,   p a d d i n g = ' s a m e ' ) )  
         m o d e l 3 . a d d ( t f . k e r a s . l a y e r s . B a t c h N o r m a l i z a t i o n ( ) )  
         m o d e l 3 . a d d ( M a x P o o l i n g 2 D ( ( 2 ,   2 ) ) )  
         m o d e l 3 . a d d ( D r o p o u t ( 0 . 2 ) )  
         m o d e l 3 . a d d ( C o n v 2 D ( 1 2 8 ,   k e r n e l _ s i z e = ( 2 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 1 ,   p a d d i n g = ' s a m e ' ) )  
         m o d e l 3 . a d d ( C o n v 2 D ( 1 2 8 ,   k e r n e l _ s i z e = 1 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
         m o d e l 3 . a d d ( F l a t t e n ( ) )  
         m o d e l 3 . a d d ( D e n s e ( 1 2 8 ,   a c t i v a t i o n = ' t a n h ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
         m o d e l 3 . a d d ( D r o p o u t ( 0 . 2 ) )  
         m o d e l 3 . a d d ( D e n s e ( 6 4 ,   a c t i v a t i o n = ' t a n h ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
         m o d e l 3 . a d d ( D e n s e ( 5 0 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
         m o d e l 3 . a d d ( D e n s e ( 1 0 ,   a c t i v a t i o n = ' s o f t m a x ' ) )  
         m o d e l 3 . c o m p i l e ( o p t i m i z e r = ' a d a m ' ,   l o s s = ' c a t e g o r i c a l _ c r o s s e n t r o p y ' ,   m e t r i c s = [ ' a c c u r a c y ' ] )  
         r e t u r n   m o d e l 3  
  
  
  
 d e f   V G G _ 2 ( ) :  
         m o d e l 4   =   S e q u e n t i a l ( )  
         m o d e l 4 . a d d ( C o n v 2 D ( 3 2 ,   ( 5 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   d a t a _ f o r m a t = ' c h a n n e l s _ l a s t ' ,   p a d d i n g = ' s a m e ' ,   i n p u t _ s h a p e = ( 6 4 ,   3 2 ,   3 ) ) )  
         m o d e l 4 . a d d ( C o n v 2 D ( 3 2 ,   ( 5 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 2 ,   p a d d i n g = ' s a m e ' ) )  
         m o d e l 4 . a d d ( t f . k e r a s . l a y e r s . B a t c h N o r m a l i z a t i o n ( ) )  
         m o d e l 4 . a d d ( A v e r a g e P o o l i n g 2 D ( ( 3 ,   3 ) ) )  
         m o d e l 4 . a d d ( C o n v 2 D ( 6 4 ,   ( 3 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 1 ,   p a d d i n g = ' s a m e ' ) )  
         m o d e l 4 . a d d ( C o n v 2 D ( 6 4 ,   ( 3 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 1 ,   p a d d i n g = ' s a m e ' ) )  
         m o d e l 4 . a d d ( t f . k e r a s . l a y e r s . B a t c h N o r m a l i z a t i o n ( ) )  
         m o d e l 4 . a d d ( M a x P o o l i n g 2 D ( ( 2 ,   2 ) ) )  
         m o d e l 4 . a d d ( C o n v 2 D ( 1 2 8 ,   ( 2 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
         m o d e l 4 . a d d ( F l a t t e n ( ) )  
         m o d e l 4 . a d d ( t f . k e r a s . l a y e r s . B a t c h N o r m a l i z a t i o n ( ) )  
         m o d e l 4 . a d d ( D e n s e ( 1 2 8 ,   a c t i v a t i o n = ' t a n h ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
         m o d e l 4 . a d d ( D e n s e ( 6 4 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
         m o d e l 4 . a d d ( D e n s e ( 1 0 ,   a c t i v a t i o n = ' s o f t m a x ' ) )  
         m o d e l 4 . c o m p i l e ( o p t i m i z e r = ' a d a m ' ,   l o s s = ' c a t e g o r i c a l _ c r o s s e n t r o p y ' ,   m e t r i c s = [ ' a c c u r a c y ' ] )  
         r e t u r n   m o d e l 4  
  
 d e f   V G G _ 3 ( ) :  
     m o d e l 5   =   S e q u e n t i a l ( )  
     m o d e l 5 . a d d ( C o n v 2 D ( 3 2 ,   ( 7 ,   5 ) ,   a c t i v a t i o n = ' r e l u ' ,   d a t a _ f o r m a t = ' c h a n n e l s _ l a s t ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ,   i n p u t _ s h a p e = ( 6 4 ,   3 2 ,   3 ) ) )  
     m o d e l 5 . a d d ( C o n v 2 D ( 6 4 ,   ( 7 ,   5 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 1 ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 5 . a d d ( M a x P o o l i n g 2 D ( ( 2 ,   2 ) ) )  
     m o d e l 5 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 5 . a d d ( C o n v 2 D ( 6 4 ,   ( 5 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 1 ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 5 . a d d ( C o n v 2 D ( 1 2 8 ,   ( 5 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   s t r i d e s = 1 ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 5 . a d d ( M a x P o o l i n g 2 D ( ( 2 ,   2 ) ) )  
     m o d e l 5 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 5 . a d d ( C o n v 2 D ( 1 2 8 ,   ( 3 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 5 . a d d ( C o n v 2 D ( 1 2 8 ,   ( 3 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 5 . a d d ( M a x P o o l i n g 2 D ( ( 3 ,   3 ) ) )  
     m o d e l 5 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 5 . a d d ( F l a t t e n ( ) )  
     m o d e l 5 . a d d ( D e n s e ( 1 2 8 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
     m o d e l 5 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 5 . a d d ( D e n s e ( 6 4 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
     m o d e l 5 . a d d ( D e n s e ( 1 0 ,   a c t i v a t i o n = ' s o f t m a x ' ) )  
     m o d e l 5 . c o m p i l e ( o p t i m i z e r = ' a d a m ' ,   l o s s = ' c a t e g o r i c a l _ c r o s s e n t r o p y ' ,   m e t r i c s = [ ' a c c u r a c y ' ] )  
     r e t u r n   m o d e l 5  
  
 d e f   d e f i n e _ m o d e l _ A l e x n e t ( ) :  
     m o d e l 1   =   S e q u e n t i a l ( )  
     m o d e l 1 . a d d ( C o n v 2 D ( 9 6 ,   ( 9 ,   7 ) ,   a c t i v a t i o n = ' r e l u ' ,   d a t a _ f o r m a t = ' c h a n n e l s _ l a s t ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ,   i n p u t _ s h a p e = ( 6 4 ,   3 2 ,   3 ) ) )  
     m o d e l 1 . a d d ( M a x P o o l i n g 2 D ( ( 3 ,   3 ) ) )  
     m o d e l 1 . a d d ( C o n v 2 D ( 2 5 6 ,   ( 7 ,   5 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 1 . a d d ( M a x P o o l i n g 2 D ( ( 3 ,   3 ) ) )  
     m o d e l 1 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 1 . a d d ( C o n v 2 D ( 3 8 4 ,   ( 5 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 1 . a d d ( C o n v 2 D ( 2 5 6 ,   ( 5 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 1 . a d d ( M a x P o o l i n g 2 D ( ( 3 ,   3 ) ) )  
     m o d e l 1 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 1 . a d d ( F l a t t e n ( ) )  
     m o d e l 1 . a d d ( D e n s e ( 1 2 8 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
     m o d e l 1 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 1 . a d d ( D e n s e ( 6 4 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
     m o d e l 1 . a d d ( D e n s e ( 1 0 ,   a c t i v a t i o n = ' s o f t m a x ' ) )  
     m o d e l 1 . c o m p i l e ( o p t i m i z e r = ' a d a m ' ,   l o s s = ' c a t e g o r i c a l _ c r o s s e n t r o p y ' ,   m e t r i c s = [ ' a c c u r a c y ' ] )  
     r e t u r n   m o d e l 1  
 d e f   m o d e l _ N e w ( ) :  
     m o d e l 2   =   S e q u e n t i a l ( )  
     m o d e l 2 . a d d ( C o n v 2 D ( 4 8 ,   ( 5 ,   5 ) ,   a c t i v a t i o n = ' r e l u ' ,   d a t a _ f o r m a t = ' c h a n n e l s _ l a s t ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ,   i n p u t _ s h a p e = ( 6 4 ,   3 2 ,   3 ) ) )  
     m o d e l 2 . a d d ( M a x P o o l i n g 2 D ( ( 2 ,   2 ) ) )  
     m o d e l 2 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 2 . a d d ( C o n v 2 D ( 6 4 ,   ( 5 ,   5 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 2 . a d d ( M a x P o o l i n g 2 D ( ( 2 ,   2 ) ) )  
     m o d e l 2 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 2 . a d d ( C o n v 2 D ( 1 2 8 ,   ( 3 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 2 . a d d ( M a x P o o l i n g 2 D ( ( 2 ,   2 ) ) )  
     m o d e l 2 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 2 . a d d ( C o n v 2 D ( 1 6 0 ,   ( 3 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 2 . a d d ( C o n v 2 D ( 1 6 0 ,   ( 3 ,   3 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 2 . a d d ( M a x P o o l i n g 2 D ( ( 2 ,   2 ) ) )  
     m o d e l 2 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 2 . a d d ( C o n v 2 D ( 1 9 2 ,   ( 2 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 2 . a d d ( C o n v 2 D ( 1 9 2 ,   ( 2 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 2 . a d d ( C o n v 2 D ( 1 9 2 ,   ( 2 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 2 . a d d ( C o n v 2 D ( 1 9 2 ,   ( 2 ,   2 ) ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ,   p a d d i n g = ' s a m e ' ) )  
     m o d e l 2 . a d d ( M a x P o o l i n g 2 D ( ( 2 ,   2 ) ) )  
     m o d e l 2 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 2 . a d d ( F l a t t e n ( ) )  
     m o d e l 2 . a d d ( D e n s e ( 1 8 0 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
     m o d e l 2 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 2 . a d d ( D e n s e ( 1 0 0 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
     m o d e l 2 . a d d ( D r o p o u t ( 0 . 2 ) )  
     m o d e l 2 . a d d ( D e n s e ( 6 0 ,   a c t i v a t i o n = ' r e l u ' ,   k e r n e l _ i n i t i a l i z e r = ' h e _ u n i f o r m ' ) )  
     m o d e l 2 . a d d ( D e n s e ( 1 0 ,   a c t i v a t i o n = ' s o f t m a x ' ) )  
     m o d e l 2 . c o m p i l e ( o p t i m i z e r = ' a d a m ' ,   l o s s = ' c a t e g o r i c a l _ c r o s s e n t r o p y ' ,   m e t r i c s = [ ' a c c u r a c y ' ] )  
     r e t u r n   m o d e l 2  
  
  
